"""
segment_sorted_units_by_condition.py
====================================

Segments globally-sorted Kilosort4 NPRW single units by fine-grained experimental
condition and target (Target A vs Target B vs At-Rest), aligned to t = 0 ms event onset.

Key Features & Methodological Alignment:
  1. Unit ID Permanence Across Session:
     - Preserves identical unit IDs [u_1, u_2, ..., u_N] across all exported files,
       ensuring cross-condition and cross-target unit comparability.
  2. Hardware Clock Drift & Shift Alignment:
     - Maps Intan spike sample indices to Blackrock millisecond timeline using
       br_to_intan_shifts.csv and rcp.intan_samples_to_br_ms with PPM correction.
  3. Fine-Grained Condition Discovery (matching plot_behavioral_rsa & plot_lfp_rsa):
     - Discovers and categorizes conditions beyond simple "control vs stim":
       * Control reaches (no stimulation)
       * Stim parameter conditions (e.g., 400Hz-2.5ms, 400Hz-20ms, 130Hz-100ms)
       * Continuous stimulation (e.g., 2Hz continuous)
       * Current steering / channel subsets
       * At-Rest conditions (no reach, stim delivered at rest)
  4. Target Sub-Division:
     - Reaching conditions are segregated into Target A (left) and Target B (right).
     - At-Rest conditions are grouped with Target = None ("At-Rest").
  5. Event Alignment (t = 0 ms):
     - Reaching reaches: aligned to IR beam crossing (which is also stim initiation for stim trials).
     - At-Rest trials: aligned to stimulation pulse/train onset.
  6. Manual Trial Exclusion:
     - Drops corrupted/invalid trials defined in config/manual_trial_remove.csv.
  7. Kinematic Integration:
     - Extracts Middle Finger 3D tracking (x, y, z), instantaneous speed, and normalized
       forward distance on the exact same relative time base.
  8. Structured Checkpoint Export:
     - Exports structured .npz and .mat files to results/checkpoints/spikesorting/PeriStim_sorted/.
"""

from __future__ import annotations

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.io import savemat

# Add repository root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import RCP_analysis as rcp

try:
    from RCP_analysis.python.functions.config_loading import (
        PARAMS,
        OUT_BASE,
        METADATA_CSV,
        METADATA_ROOT,
        PERI_ROOT,
        ALIGNED_CKPT_ROOT,
        SESSION_LOC,
        REPO_ROOT,
    )
    CONFIG_LOADED = True
except Exception as _e:
    print(f"[segment_sorted] Warning: config_loading failed: {_e}")
    CONFIG_LOADED = False
    PARAMS = None
    OUT_BASE = None
    METADATA_CSV = None
    METADATA_ROOT = None
    PERI_ROOT = None
    ALIGNED_CKPT_ROOT = None
    SESSION_LOC = None
    REPO_ROOT = _REPO_ROOT


def _log(msg: str) -> None:
    print(f"[segment_sorted] {msg}", flush=True)


def load_manual_remove_trials(manual_remove_csv: Path) -> Dict[Tuple[str, int], set[int]]:
    """
    Parse config/manual_trial_remove.csv into a dictionary: (intan_filename, br_idx) -> set of dropped trial indices.
    """
    drops: Dict[Tuple[str, int], set[int]] = {}
    if not manual_remove_csv.exists():
        return drops

    try:
        df = pd.read_csv(manual_remove_csv)
        for _, row in df.iterrows():
            sess = str(row.get("intan_filename", "")).strip()
            br_val = row.get("br_idx")
            trials_str = str(row.get("trials_to_drop", "")).strip()

            if not sess or pd.isna(br_val):
                continue
            try:
                b_idx = int(br_val)
            except Exception:
                continue

            drop_set = set()
            if trials_str and trials_str.lower() not in ("nan", "none", ""):
                # Parse comma or space separated numbers
                for part in trials_str.replace(";", ",").split(","):
                    p = part.strip().strip("'\"")
                    if p.isdigit():
                        drop_set.add(int(p))

            key = (sess, b_idx)
            if key not in drops:
                drops[key] = set()
            drops[key].update(drop_set)
    except Exception as e:
        _log(f"Warning parsing manual trial removal CSV: {e}")

    return drops


def load_br_to_intan_shifts(shifts_csv: Path) -> Dict[int, Dict[str, float]]:
    """
    Parse br_to_intan_shifts.csv into a dictionary: br_idx -> shift parameters.
    """
    shifts: Dict[int, Dict[str, float]] = {}
    if not shifts_csv.exists():
        return shifts

    try:
        df = pd.read_csv(shifts_csv)
        for _, row in df.iterrows():
            br_val = row.get("br_idx")
            if pd.isna(br_val):
                continue
            try:
                b_idx = int(br_val)
            except Exception:
                continue

            shift_sample = float(row.get("shift_sample", row.get("adjusted_shift_sample", 0.0)))
            shift_ms = float(row.get("shift_ms", 0.0))
            shifts[b_idx] = {
                "shift_sample": shift_sample,
                "shift_ms": shift_ms,
                "dur_intan_sec": float(row.get("dur_intan_sec", 0.0)),
                "dur_br_sec": float(row.get("dur_br_sec", 0.0)),
            }
    except Exception as e:
        _log(f"Warning reading shifts CSV: {e}")

    return shifts


def classify_condition_and_label(
    row_meta: Dict[str, Any],
    aligned_meta: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Construct condition type ('control_reach', 'stim_reach', 'at_rest', 'continuous_stim')
    and fine-grained descriptive label matching plot_behavioral_rsa & plot_lfp_rsa.
    """
    notes = str(row_meta.get("Notes", row_meta.get("notes", ""))).strip().lower()
    movement_trigger = str(row_meta.get("Movement_Trigger", row_meta.get("movement_trigger", ""))).strip().lower()

    # Stimulation parameters
    freq_hz = row_meta.get("Frequency", row_meta.get("stim_freq_hz", aligned_meta.get("stim_freq_hz")))
    pulses = row_meta.get("Pulses", row_meta.get("stim_pulse_count", aligned_meta.get("stim_pulse_counts")))
    dur_ms = row_meta.get("Duration", row_meta.get("stim_dur_nominal_ms", aligned_meta.get("stim_dur_nominal_ms")))
    channels = row_meta.get("Channels", row_meta.get("stim_channels", aligned_meta.get("stim_channels", "")))

    try:
        freq_hz = float(freq_hz) if pd.notna(freq_hz) else None
    except Exception:
        freq_hz = None

    try:
        if isinstance(pulses, (list, np.ndarray)) and len(pulses) > 0:
            pulses = int(pulses[0])
        else:
            pulses = int(pulses) if pd.notna(pulses) else None
    except Exception:
        pulses = None

    try:
        dur_ms = float(dur_ms) if pd.notna(dur_ms) else None
    except Exception:
        dur_ms = None

    is_baseline = (notes == "baseline" or "control" in notes or (freq_hz is None and pulses is None))
    is_at_rest = (notes == "rest" or "rest" in notes or movement_trigger == "rest")
    is_sig_gen = (movement_trigger == "sig gen")
    is_continuous = (is_sig_gen and not is_at_rest) or (freq_hz is not None and freq_hz <= 2.5)

    cond_params = {
        "stim_freq_hz": freq_hz,
        "stim_pulse_count": pulses,
        "stim_dur_nominal_ms": dur_ms,
        "stim_channels": str(channels).strip() if pd.notna(channels) else "",
        "notes": notes,
        "movement_trigger": movement_trigger,
    }

    if is_baseline:
        cond_type = "control_reach"
        cond_label = "Control"
    elif is_at_rest:
        cond_type = "at_rest"
        label_parts = ["At-Rest"]
        if freq_hz is not None:
            label_parts.append(f"{int(round(freq_hz))}Hz")
        if pulses is not None:
            label_parts.append(f"{pulses}p")
        elif dur_ms is not None:
            label_parts.append(f"{dur_ms:.1f}ms")
        cond_label = "_".join(label_parts)
    elif is_continuous:
        cond_type = "continuous_stim"
        cond_label = f"Continuous_{int(round(freq_hz)) if freq_hz else 2}Hz"
    else:
        cond_type = "stim_reach"
        label_parts = []
        if freq_hz is not None:
            label_parts.append(f"{int(round(freq_hz))}Hz")
        if pulses is not None:
            label_parts.append(f"{pulses}p")
        if dur_ms is not None and (pulses is None or pulses > 1):
            label_parts.append(f"{dur_ms:.1f}ms")
        if "steer" in notes or "curr steer" in notes:
            label_parts.append("Steer")
        cond_label = "_".join(label_parts) if label_parts else "Stim"

    return cond_type, cond_label, cond_params




def run_segmentation(
    session_loc: Optional[Path] = None,
    time_window_ms: Tuple[float, float] = (-500.0, 500.0),
    ppm_correction: float = -13.951,
) -> Path:
    """
    Segment globally-sorted units into per-condition, per-target datasets.
    """
    if session_loc is None:
        if SESSION_LOC is None:
            raise ValueError("SESSION_LOC must be specified or configured in params.yaml.")
        session_loc = Path(SESSION_LOC)

    out_base = session_loc / "results"


    spikesorting_root = out_base / "checkpoints" / "spikesorting"
    concat_ckpt = spikesorting_root / "NPRW_concat_blanking"

    sess_index_file = concat_ckpt / "session_index.pkl"
    curated_folder = concat_ckpt / "sorting_curated"

    if not sess_index_file.exists():
        raise FileNotFoundError(f"Session index pkl not found at: {sess_index_file}. Run sort_nprw_kilosort4.py first.")

    with open(sess_index_file, "rb") as f:
        sess_index = pickle.load(f)

    session_boundaries = sess_index["session_boundaries"]
    sess_names = sess_index["sess_names"]
    br_indices = sess_index["br_indices"]
    fs_intan = float(sess_index.get("fs", 30000.0))

    _log(f"Loaded session index: {len(sess_names)} conditions, total samples: {session_boundaries[-1]}")

    # Load manually curated sorting or per-session spike trains
    import spikeinterface as si
    import spikeinterface.extractors as se

    sorting = None
    manual_analyzer = concat_ckpt / "analyzer_manual"
    curated_folder = concat_ckpt / "sorting_curated"
    phy_curated_folder = concat_ckpt / "phy_for_curation"
    slay_folder = concat_ckpt / "analyzer_SLAy"

    if manual_analyzer.exists():
        _log(f"Loading manually curated SortingAnalyzer from: {manual_analyzer}")
        analyzer_manual = si.load_sorting_analyzer(manual_analyzer)
        sorting = analyzer_manual.sorting
        unit_ids = np.asarray(sorting.unit_ids)
    elif curated_folder.exists():
        _log(f"Loading curated SpikeInterface sorting from: {curated_folder}")
        sorting = si.load(curated_folder)
        unit_ids = np.asarray(sorting.unit_ids)
    elif phy_curated_folder.exists() and ((phy_curated_folder / "cluster_group.tsv").exists() or (phy_curated_folder / "cluster_info.tsv").exists()):
        _log(f"Loading manually curated Phy sorting directly from: {phy_curated_folder}")
        sorting = se.read_phy(phy_curated_folder, exclude_cluster_groups=["noise"])
        unit_ids = np.asarray(sorting.unit_ids)
    elif slay_folder.exists():
        _log(f"Fallback: Loading legacy post-SLAY curated sorting from: {slay_folder}")
        analyzer_slay = si.load_sorting_analyzer(slay_folder)
        sorting = analyzer_slay.sorting
        unit_ids = np.asarray(sorting.unit_ids)
    else:
        # Fall back to first session spike_trains.npz to inspect unit_ids
        spk_sample = spikesorting_root / "by_session" / sess_names[0] / "NPRW" / "spike_trains.npz"
        if not spk_sample.exists():
            raise FileNotFoundError(f"Neither analyzer_manual, sorting_curated, phy_for_curation, nor per-session spike_trains found at: {concat_ckpt}")
        npz_spk = np.load(spk_sample)
        unit_keys = [k for k in npz_spk.files if k.startswith("unit_")]
        unit_ids = np.array([int(k.replace("unit_", "")) for k in sorted(unit_keys, key=lambda x: int(x.replace("unit_", "")))])

    _log(f"Identified {len(unit_ids)} tracked IntA units across session.")

    # Load manual trial remove CSV
    manual_remove_csv = REPO_ROOT / "config" / "manual_trial_remove.csv"
    manual_drops = load_manual_remove_trials(manual_remove_csv)

    # Load shifts CSV
    shifts_csv = session_loc / "Metadata" / "br_to_intan_shifts.csv"
    if not shifts_csv.exists() and METADATA_ROOT:
        shifts_csv = METADATA_ROOT / "br_to_intan_shifts.csv"
    shifts_dict = load_br_to_intan_shifts(shifts_csv)

    # Load metadata CSV
    meta_csv = None
    if METADATA_CSV and Path(METADATA_CSV).exists():
        meta_csv = Path(METADATA_CSV)
    else:
        meta_cands = [f for f in (session_loc / "Metadata").glob("*metadata*.csv") if not f.name.startswith("._")]
        if meta_cands:
            meta_csv = meta_cands[0]

    meta_rows: Dict[int, Dict[str, Any]] = {}
    if meta_csv and meta_csv.exists():
        try:
            df_m = pd.read_csv(meta_csv, encoding="latin1")
            for _, r in df_m.iterrows():
                b_val = r.get("BR_File", r.get("br_idx"))
                if pd.notna(b_val):
                    try:
                        meta_rows[int(b_val)] = r.to_dict()
                    except Exception:
                        pass
        except Exception as e:
            _log(f"Warning reading metadata CSV: {e}")

    # Output directory for PeriStim_sorted under checkpoints/spikesorting/
    spikesorting_root = out_base / "checkpoints" / "spikesorting"
    peri_sorted_root = spikesorting_root / "PeriStim_sorted"
    peri_sorted_root.mkdir(parents=True, exist_ok=True)


    # Process each condition
    _log("Step 2: Processing conditions and segmenting spikes by target...")
    total_files_exported = 0

    for i, (s_name, b_idx) in enumerate(zip(sess_names, br_indices)):
        t_start = int(session_boundaries[i])
        t_end = int(session_boundaries[i + 1])

        # Shift sample and clock alignment
        shift_info = shifts_dict.get(b_idx, {})
        shift_sample = shift_info.get("shift_sample", 0.0)

        # 1. Retrieve Intan spike trains for this session
        session_spike_trains: Dict[int, np.ndarray] = {}
        if sorting is not None:
            for uid in unit_ids:
                st = sorting.get_unit_spike_train(uid, segment_index=0)
                mask = (st >= t_start) & (st < t_end)
                session_spike_trains[int(uid)] = st[mask] - t_start
        else:
            spk_file = spikesorting_root / "by_session" / s_name / "NPRW" / "spike_trains.npz"
            npz_spk = np.load(spk_file)
            for uid in unit_ids:
                session_spike_trains[int(uid)] = npz_spk.get(f"unit_{uid}", np.zeros(0, int))

        # Convert spike times from Intan samples to Blackrock milliseconds
        session_spike_br_ms: Dict[int, np.ndarray] = {}
        for uid, samps in session_spike_trains.items():
            if samps.size > 0:
                ms = rcp.intan_samples_to_br_ms(samps, shift_sample, fs_intan=fs_intan, ppm_correction=ppm_correction)
                session_spike_br_ms[uid] = np.asarray(ms, float)
            else:
                session_spike_br_ms[uid] = np.zeros(0, float)

        # 2. Discover trials and event times from Aligned or PeriStim
        aligned_file = None
        aligned_candidates = list((out_base / "checkpoints" / "Aligned").rglob(f"*__{s_name}__*BR_{b_idx:03d}*.npz"))
        if aligned_candidates:
            aligned_file = aligned_candidates[0]

        peri_candidates = list((out_base / "checkpoints" / "PeriStim").rglob(f"*__{s_name}__BR_{b_idx:03d}*.npz"))
        if not peri_candidates:
            peri_candidates = list((out_base / "checkpoints" / "PeriStim").rglob(f"*__BR_{b_idx:03d}*.npz"))

        row_meta = meta_rows.get(b_idx, {})

        # Collect all trial streams for this recording block (can have target_A, target_B, and/or at_rest)
        streams_to_process: List[Dict[str, Any]] = []

        if peri_candidates:
            for p_cand in peri_candidates:
                if p_cand.name.startswith("._"):
                    continue
                try:
                    p_npz = np.load(p_cand, allow_pickle=True)
                    if "event_ms" not in p_npz or p_npz["event_ms"].size == 0:
                        continue
                    ev_ms = np.asarray(p_npz["event_ms"], float)

                    # Determine trial labels
                    tr_lbls = np.asarray(p_npz["trial_labels"], str) if "trial_labels" in p_npz else None
                    if tr_lbls is None or len(tr_lbls) != len(ev_ms):
                        if "target_a" in p_cand.name.lower() or p_cand.parent.name == "target_A":
                            tr_lbls = np.repeat("A", len(ev_ms))
                        elif "target_b" in p_cand.name.lower() or p_cand.parent.name == "target_B":
                            tr_lbls = np.repeat("B", len(ev_ms))
                        else:
                            tr_lbls = np.repeat("None", len(ev_ms))

                    # Determine target tag
                    tgt_tag = "None"
                    if "target_a" in p_cand.name.lower() or p_cand.parent.name == "target_A" or np.all(np.char.upper(tr_lbls) == "A"):
                        tgt_tag = "target_A"
                    elif "target_b" in p_cand.name.lower() or p_cand.parent.name == "target_B" or np.all(np.char.upper(tr_lbls) == "B"):
                        tgt_tag = "target_B"
                    elif "at_rest" in str(p_cand).lower() or "none" in p_cand.name.lower():
                        tgt_tag = "None"

                    # Kinematics
                    b_rel_t = p_npz.get("beh_rel_t")
                    cam1_pos = p_npz.get("beh_cam1_segs")
                    cam1_vel = p_npz.get("beh_cam1_vel_segs")
                    cam1_names = p_npz.get("beh_cam1_names")
                    if cam1_pos is None or (isinstance(cam1_pos, np.ndarray) and cam1_pos.size == 0):
                        cam1_pos = p_npz.get("beh_cam0_segs")
                        cam1_vel = p_npz.get("beh_cam0_vel_segs")
                        cam1_names = p_npz.get("beh_cam0_names")

                    norm_dist = None
                    speed_arr = None
                    if cam1_pos is not None and b_rel_t is not None and rcp is not None:
                        try:
                            names_l = [str(x) for x in np.asarray(cam1_names).ravel()] if cam1_names is not None else []
                            raw_l = [rcp.strip_cam_prefix(n) for n in names_l]
                            disp_l = rcp.simple_beh_labels(raw_l, ("middle", "wrist"))
                            norm_dist = rcp.compute_normalized_distance_from_xy_3d(
                                cam1_pos, disp_l, b_rel_t, keypoint="middle", ref_time_ms=-600.0, mode="max_abs"
                            )
                            idx_x, idx_y = rcp.find_xy_indices_for_keypoint(disp_l, "middle")
                            if idx_x is not None and idx_y is not None and cam1_vel is not None and cam1_vel.ndim == 3:
                                speed_arr = np.sqrt(cam1_vel[:, idx_x, :]**2 + cam1_vel[:, idx_y, :]**2)
                        except Exception:
                            pass

                    al_meta = {}
                    if "meta" in p_npz:
                        al_meta = p_npz["meta"].item() if hasattr(p_npz["meta"], "item") else dict(p_npz["meta"])

                    streams_to_process.append({
                        "events_ms": ev_ms,
                        "trial_labels": tr_lbls,
                        "target_tag": tgt_tag,
                        "norm_pos": norm_dist,
                        "speed": speed_arr,
                        "beh_rel_t": b_rel_t,
                        "aligned_meta": al_meta,
                    })
                except Exception as e:
                    _log(f"  Warning loading {p_cand.name}: {e}")

        # If no PeriStim candidates, fallback to Aligned npz
        if not streams_to_process and aligned_file is not None:
            try:
                a_npz = np.load(aligned_file, allow_pickle=True)
                ir_ms = np.asarray(a_npz["ir_ms"], float) if "ir_ms" in a_npz else np.zeros(0)
                stim_ms = np.asarray(a_npz["stim_ms"], float) if "stim_ms" in a_npz else np.zeros(0)
                ev_ms = ir_ms if ir_ms.size > 0 else stim_ms
                if len(ev_ms) > 0:
                    if "ts_state_char" in a_npz and len(a_npz["ts_state_char"]) == len(ev_ms):
                        tr_lbls = np.asarray(a_npz["ts_state_char"], str)
                    else:
                        tr_lbls = np.repeat("A", len(ev_ms))
                    al_meta = a_npz["align_meta"].item() if hasattr(a_npz["align_meta"], "item") else dict(a_npz.get("align_meta", {}))
                    streams_to_process.append({
                        "events_ms": ev_ms,
                        "trial_labels": tr_lbls,
                        "target_tag": "multi",
                        "norm_pos": None,
                        "speed": None,
                        "beh_rel_t": None,
                        "aligned_meta": al_meta,
                    })
            except Exception:
                pass

        if not streams_to_process:
            _log(f"  [{s_name}] (BR_{b_idx:03d}): No event timestamps found. Skipping.")
            continue

        # Slicing window
        w_min, w_max = time_window_ms

        # Process each stream (target_A, target_B, or at_rest)
        for stream in streams_to_process:
            events_ms = stream["events_ms"]
            trial_labels = stream["trial_labels"]
            aligned_meta = stream["aligned_meta"]
            tgt_tag = stream["target_tag"]
            kinematics_segs = stream["norm_pos"]
            speed_arr = stream["speed"]
            beh_rel_t = stream["beh_rel_t"]

            # Condition classification
            cond_type, cond_label, cond_params = classify_condition_and_label(row_meta, aligned_meta)
            _log(f"  [{s_name}] BR_{b_idx:03d} -> type='{cond_type}', label='{cond_label}', stream='{tgt_tag}', trials={len(events_ms)}")

            # Check manual trial drops
            drops = manual_drops.get((s_name, b_idx), set())
            valid_mask = np.array([tr not in drops for tr in range(len(events_ms))], dtype=bool)

            if not np.any(valid_mask):
                _log(f"  [{s_name}] All trials manually dropped. Skipping.")
                continue

            # Determine target groups to export
            if cond_type == "at_rest":
                target_groups = [("None", np.flatnonzero(valid_mask))]
            elif tgt_tag in ("target_A", "target_B"):
                target_groups = [(tgt_tag, np.flatnonzero(valid_mask))]
            else:
                idx_A = np.flatnonzero(valid_mask & (np.char.upper(trial_labels) == "A"))
                idx_B = np.flatnonzero(valid_mask & (np.char.upper(trial_labels) == "B"))
                target_groups = []
                if len(idx_A) > 0:
                    target_groups.append(("target_A", idx_A))
                if len(idx_B) > 0:
                    target_groups.append(("target_B", idx_B))
                if len(target_groups) == 0:
                    target_groups.append(("target_A", np.flatnonzero(valid_mask)))

            for tgt_name, tr_indices in target_groups:
                n_sub_trials = len(tr_indices)
                sub_events = events_ms[tr_indices]

                # trial_spikes: (n_trials, n_units) object array of relative spike times in ms
                trial_spikes_matrix = np.empty((n_sub_trials, len(unit_ids)), dtype=object)

                for tr_local, tr_orig in enumerate(tr_indices):
                    t_ev = events_ms[tr_orig]
                    for u_idx, uid in enumerate(unit_ids):
                        all_spk = session_spike_br_ms[uid]
                        # Relative time: t_rel = t_spk - t_event
                        in_win = (all_spk >= (t_ev + w_min)) & (all_spk <= (t_ev + w_max))
                        rel_ms = all_spk[in_win] - t_ev
                        trial_spikes_matrix[tr_local, u_idx] = rel_ms.astype(np.float32)

                # Sub-directory
                if cond_type == "at_rest":
                    out_dir = peri_sorted_root / "at_rest"
                elif cond_type == "continuous_stim":
                    out_dir = peri_sorted_root / "continuous_stim"
                elif cond_type == "control_reach":
                    out_dir = peri_sorted_root / "control_reaches" / tgt_name
                else:
                    out_dir = peri_sorted_root / "stim_reaches" / tgt_name

                out_dir.mkdir(parents=True, exist_ok=True)

                file_stem = f"sorted_peristim__{s_name}__BR_{b_idx:03d}_{tgt_name}"
                out_npz = out_dir / f"{file_stem}.npz"
                out_mat = out_dir / f"{file_stem}.mat"

                # Prepare export dictionary
                to_save = {
                    "unit_ids": unit_ids,
                    "trial_spikes": trial_spikes_matrix,
                    "event_ms": sub_events,
                    "trial_indices": tr_indices,
                    "target": tgt_name.replace("target_", ""),
                    "condition_type": cond_type,
                    "condition_label": cond_label,
                    "time_window_ms": np.array([w_min, w_max], dtype=np.float32),
                    "fs_intan": fs_intan,
                    "session_name": s_name,
                    "br_idx": b_idx,
                }
                to_save.update(cond_params)

                # Kinematics if available (matching plot_firing_rates.py)
                if kinematics_segs is not None and len(kinematics_segs) == len(events_ms):
                    to_save["norm_pos"] = kinematics_segs[tr_indices]
                    to_save["kinematics_segs"] = kinematics_segs[tr_indices]
                if speed_arr is not None and len(speed_arr) == len(events_ms):
                    to_save["speed"] = speed_arr[tr_indices]
                if beh_rel_t is not None:
                    to_save["beh_rel_t"] = beh_rel_t

                np.savez_compressed(out_npz, **to_save)

                # MATLAB .mat export
                mat_dict = {
                    "unit_ids": unit_ids.astype(np.int64),
                    "event_ms": sub_events.astype(np.float64),
                    "trial_indices": tr_indices.astype(np.int64),
                    "target": tgt_name,
                    "condition_type": cond_type,
                    "condition_label": cond_label,
                    "time_window_ms": np.array([w_min, w_max], dtype=np.float64),
                    "fs_intan": float(fs_intan),
                    "session_name": s_name,
                    "br_idx": int(b_idx),
                }
                # Add spike trains per unit for MATLAB convenience: unit_{uid}
                for u_idx, uid in enumerate(unit_ids):
                    spk_cells = np.empty((n_sub_trials, 1), dtype=object)
                    for tr in range(n_sub_trials):
                        spk_cells[tr, 0] = trial_spikes_matrix[tr, u_idx].astype(np.float64)
                    mat_dict[f"unit_{uid}"] = spk_cells

                if kinematics_segs is not None and len(kinematics_segs) == len(events_ms):
                    mat_dict["norm_pos"] = np.asarray(kinematics_segs[tr_indices], np.float64)
                if speed_arr is not None and len(speed_arr) == len(events_ms):
                    mat_dict["speed"] = np.asarray(speed_arr[tr_indices], np.float64)
                if beh_rel_t is not None:
                    mat_dict["beh_rel_t"] = np.asarray(beh_rel_t, np.float64)

                savemat(out_mat, mat_dict, do_compression=True)
                total_files_exported += 1

            savemat(out_mat, mat_dict, do_compression=True)
            _log(f"  [SAVED] {tgt_name}: {n_sub_trials} trials -> {out_npz.name}")
            total_files_exported += 1

    # Update pipeline status
    try:
        from RCP_analysis.python.functions.utils import update_status_cell
        update_status_cell(PARAMS.monkey, PARAMS.session, "segment_sorted", "done")
        _log("Updated data_status_reaching.csv: segment_sorted = done")
    except Exception as e:
        _log(f"Status update note: {e}")

    _log("=" * 70)
    _log(f"Condition & Target Segmentation Complete: {total_files_exported} files exported!")
    _log(f"Checkpoints saved to: {peri_sorted_root}")
    _log("=" * 70)
    return peri_sorted_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment sorted NPRW units by condition and target.")
    parser.add_argument("--session-dir", type=str, default=None, help="Path to session directory.")
    parser.add_argument("--win-min", type=float, default=-500.0, help="Window start ms relative to event.")
    parser.add_argument("--win-max", type=float, default=500.0, help="Window end ms relative to event.")
    args = parser.parse_args()

    sess_dir = Path(args.session_dir) if args.session_dir else None
    run_segmentation(session_loc=sess_dir, time_window_ms=(args.win_min, args.win_max))


if __name__ == "__main__":
    main()

