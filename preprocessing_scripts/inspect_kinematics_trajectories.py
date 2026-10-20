"""
Plot Kinematics Trajectories for Trial Inspection And Manual Removal
====================================================================
Subplot titles show:
  - "Trial:i (Raw:R)" where:
    - i = position in this plot (for reference)
    - R = stable raw stim-pulse index -> THIS is what goes in manual_trial_remove.csv
  - Subplots with a red highlight (thick red border, reddish background, and red title)
    represent trials that have been removed (either via manual_trial_remove.csv
    or via automatic kinematic filtering / gating).
"""

import math
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from RCP_analysis.python.functions.config_loading import *

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
N_COLS = 6
FIG_WIDTH_PER_COL = 2.5
FIG_HEIGHT_PER_ROW = 2.6
CAMERA = "cam1"  # Which camera to use
KEYPOINT = "middle"  # Which keypoint (middle finger)

# Fallback root resolution if not loaded
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
MANUAL_REMOVE_CSV = (REPO_ROOT / "config" / "manual_trial_remove.csv") if "REPO_ROOT" in globals() and REPO_ROOT else (SCRIPT_REPO_ROOT / "config" / "manual_trial_remove.csv")

# ---------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------
if OUT_BASE is not None:
    FIG_ROOT = OUT_BASE / "figures" / "kinematics_trajectories"
else:
    FIG_ROOT = SCRIPT_REPO_ROOT / "results" / "figures" / "kinematics_trajectories"
FIG_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# MANUAL REMOVAL RULES LOADER
# ---------------------------------------------------------------------
def load_manual_remove_rules() -> pd.DataFrame:
    """Load manual_trial_remove.csv rules."""
    if not MANUAL_REMOVE_CSV.exists():
        return pd.DataFrame(columns=['intan_filename', 'br_idx', 'trials_to_drop', 'reason'])
    try:
        df = pd.read_csv(MANUAL_REMOVE_CSV)
        df = df.dropna(how='all')
        return df
    except Exception as e:
        print(f"[IKT] Warning reading {MANUAL_REMOVE_CSV}: {e}")
        return pd.DataFrame(columns=['intan_filename', 'br_idx', 'trials_to_drop', 'reason'])


def get_manual_removed_trials_for_session(df_rules: pd.DataFrame, intan_filename: str, br_idx: int) -> dict[int, str]:
    """
    Returns a dict of {raw_trial_idx: reason} for the given session and BR index.
    """
    if df_rules.empty:
        return {}

    csv_filenames = df_rules['intan_filename'].astype(str).str.strip()
    actual_filename = str(intan_filename).strip()
    filename_match = csv_filenames.apply(
        lambda x: len(x) > 3 and (actual_filename.startswith(x) or x in actual_filename or actual_filename in x)
    )
    br_idx_match = (
        pd.to_numeric(df_rules['br_idx'], errors='coerce').fillna(-1).astype(int) == int(br_idx)
    )
    matched_rows = df_rules[filename_match & br_idx_match]

    removed = {}
    for _, row in matched_rows.iterrows():
        if pd.notna(row.get('trials_to_drop')):
            reason = str(row.get('reason', 'manual removal')).strip()
            if not reason or reason.lower() == 'nan':
                reason = "manual removal"
            raw_str = str(row['trials_to_drop'])
            for item in raw_str.split(','):
                item_clean = item.strip()
                if item_clean.isdigit():
                    removed[int(item_clean)] = reason
    return removed


# ---------------------------------------------------------------------
# ALIGNED CHECKPOINT LOOKUP & SLICING
# ---------------------------------------------------------------------
def find_aligned_file(intan_filename: str, br_idx: int) -> Path | None:
    """Find the aligned npz checkpoint matching the session and BR index."""
    if ALIGNED_CKPT_ROOT is None or not Path(ALIGNED_CKPT_ROOT).exists():
        return None
    root = Path(ALIGNED_CKPT_ROOT)

    patterns = [
        f"**/*{intan_filename}*BR_{br_idx:03d}*.npz",
        f"**/*{intan_filename}*BR_{br_idx}*.npz",
        f"**/*BR_{br_idx:03d}*.npz",
        f"**/*BR_{br_idx}*.npz",
    ]
    for pat in patterns:
        cands = list(root.glob(pat))
        if cands:
            return cands[0]
    return None


def _compute_trial_labels(
    event_ms: np.ndarray,
    t_ms: np.ndarray,
    ts_char: np.ndarray | None,
    ts_num: np.ndarray,
    win_ms: tuple[float, float] = (0.0, 1000.0),
    *,
    num_to_label: dict[int, str] | None = None,
) -> np.ndarray:
    """Classify reach target ('A'/'B') based on touchscreen state events."""
    event_ms = np.asarray(event_ms, float).ravel()
    t_ms = np.asarray(t_ms, float).ravel()
    ts_num = np.asarray(ts_num).ravel()
    ts_char_arr = None if ts_char is None else np.asarray(ts_char).reshape(-1)

    labels = np.full(event_ms.shape, "N", dtype="U1")
    w0, w1 = map(float, win_ms)

    if t_ms.size == 0 or ts_num.size == 0 or t_ms.size != ts_num.size:
        return labels

    use_char = (ts_char_arr is not None and ts_char_arr.size == t_ms.size)

    for i, s in enumerate(event_ms):
        lo, hi = s + w0, s + w1
        i0 = np.searchsorted(t_ms, lo, side="left")
        i1 = np.searchsorted(t_ms, hi, side="right")
        if i1 <= i0:
            continue

        if use_char:
            chars = ts_char_arr[i0:i1].astype("U1")
            chars = chars[(chars == "A") | (chars == "B")]
            if chars.size:
                vals, cnts = np.unique(chars, return_counts=True)
                labels[i] = vals[np.argmax(cnts)]
        else:
            if not num_to_label:
                continue
            nums = ts_num[i0:i1].astype(int, copy=False)
            mapped = np.array([num_to_label.get(int(n), "N") for n in nums], dtype="U1")
            mapped = mapped[(mapped == "A") | (mapped == "B")]
            if mapped.size:
                vals, cnts = np.unique(mapped, return_counts=True)
                labels[i] = vals[np.argmax(cnts)]

    return labels


def find_and_extract_missing_trials(
    intan_filename: str,
    br_idx: int,
    existing_raw_indices: set[int],
    manual_rules: dict[int, str],
    target_letter: str | None,
    t_peri: np.ndarray,
    keypoint: str = "middle",
    camera: str = "cam1",
    condition_type: str = "stim",
) -> list[dict]:
    """
    Search aligned checkpoint for trials that were removed/gated out of PeriStim,
    cut their kinematics segments around stim/IR event times, and return them as trial dicts.
    """
    aligned_file = find_aligned_file(intan_filename, br_idx)
    if aligned_file is None or not aligned_file.exists():
        return []

    try:
        aligned = np.load(aligned_file, allow_pickle=True)
        cam_key = f"beh_{camera}"
        cols_key = f"beh_{camera}_cols"

        has_cam = (cam_key in aligned.files and cols_key in aligned.files)
        cam_data = aligned[cam_key] if has_cam else None
        cam_cols = [str(c) for c in aligned[cols_key]] if has_cam else []
        beh_t = aligned["beh_t_ms"] if "beh_t_ms" in aligned.files else None

        # Determine event times (stim_ms or ir_ms) based on condition
        event_ms = None
        use_ir = (condition_type in ("control", "continuous_stim"))
        if use_ir:
            if "ir_ms" in aligned.files and aligned["ir_ms"].size:
                event_ms = aligned["ir_ms"]
            elif "stim_ms" in aligned.files and aligned["stim_ms"].size:
                event_ms = aligned["stim_ms"]
        else:
            if "stim_ms" in aligned.files and aligned["stim_ms"].size:
                event_ms = aligned["stim_ms"]
            elif "ir_ms" in aligned.files and aligned["ir_ms"].size:
                event_ms = aligned["ir_ms"]

        if event_ms is None or len(event_ms) == 0:
            return []

        has_valid_beh = (
            cam_data is not None and beh_t is not None
            and hasattr(beh_t, "size") and beh_t.size > 0
            and hasattr(cam_data, "shape") and cam_data.ndim == 2 and cam_data.shape[0] > 0
        )
        idx_x, idx_y = find_xy_indices(cam_cols, keypoint) if has_valid_beh else (None, None)
        can_slice_kinematics = (has_valid_beh and idx_x is not None and idx_y is not None)

        n_events = len(event_ms)

        # Target label determination per event
        trial_targets = np.full(n_events, "N", dtype="U1")
        if target_letter is not None:
            ts_num = aligned.get("ts_state_num", None)
            ts_char = aligned.get("ts_state_char", None)
            ua_meta = aligned.get("ua_meta")
            if ua_meta is not None and hasattr(ua_meta, "item") and ua_meta.shape == ():
                ua_meta = ua_meta.item()
            if not isinstance(ua_meta, dict):
                ua_meta = {}
            rec_start_ms = float(ua_meta.get("rec_start_ms", 0.0))
            fs_ts = float(ua_meta.get("fs_ts", 1000.0))

            if ts_num is not None and np.size(ts_num):
                ts_num_arr = np.asarray(ts_num).ravel()
                ns2_t_ms = rec_start_ms + np.arange(ts_num_arr.size, dtype=float) * (1000.0 / fs_ts)
                trial_targets = _compute_trial_labels(
                    event_ms=event_ms,
                    t_ms=ns2_t_ms,
                    ts_char=ts_char,
                    ts_num=ts_num_arr,
                    win_ms=(0.0, 1000.0),
                    num_to_label={1: "A", 2: "B"},
                )

        if can_slice_kinematics:
            # Extract X and Y columns and normalize/z-score
            x_raw = cam_data[:, idx_x].astype(float)
            y_raw = cam_data[:, idx_y].astype(float)
            x_m, x_s = np.nanmean(x_raw), np.nanstd(x_raw)
            y_m, y_s = np.nanmean(y_raw), np.nanstd(y_raw)
            x_z = (x_raw - x_m) / x_s if np.isfinite(x_s) and x_s > 0 else x_raw
            y_z = (y_raw - y_m) / y_s if np.isfinite(y_s) and y_s > 0 else y_raw

        missing_trials = []
        for r_idx in range(n_events):
            # Target check
            if target_letter is not None:
                if trial_targets[r_idx] in ("A", "B"):
                    if trial_targets[r_idx] != target_letter:
                        continue
                else:
                    # Target is 'N' or unknown; only include if explicitly in manual_rules
                    if r_idx not in manual_rules:
                        continue
            else:
                # No target letter (e.g. continuous_stim, at_rest, grasp, imu):
                # Only add missing events if explicitly listed in manual_rules
                if r_idx not in manual_rules:
                    continue

            # If already present in PeriStim, skip
            if r_idx in existing_raw_indices:
                continue

            # Determine removal reason
            if r_idx in manual_rules:
                reason = manual_rules[r_idx]
            elif not can_slice_kinematics:
                reason = "no behavior tracking"
            else:
                reason = "kinematic filter"

            if can_slice_kinematics:
                t_ev = float(event_ms[r_idx])
                target_t = t_ev + t_peri

                # Interpolate onto peristim time axis
                if target_t[0] >= beh_t[0] - 100.0 and target_t[-1] <= beh_t[-1] + 100.0:
                    xi = np.interp(target_t, beh_t, x_z, left=np.nan, right=np.nan)
                    yi = np.interp(target_t, beh_t, y_z, left=np.nan, right=np.nan)
                else:
                    xi = np.full(len(t_peri), np.nan)
                    yi = np.full(len(t_peri), np.nan)
            else:
                xi = np.full(len(t_peri), np.nan)
                yi = np.full(len(t_peri), np.nan)

            missing_trials.append({
                "raw_idx": r_idx,
                "x": xi,
                "y": yi,
                "is_removed": True,
                "reason": reason,
                "source": "aligned",
            })

        return missing_trials
    except Exception as e:
        print(f"[IKT] Could not extract missing trials from aligned {aligned_file.name}: {e}")
        return []


# ---------------------------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------------------------
def find_xy_indices(names: list, keypoint: str = "middle") -> tuple:
    """Find X and Y indices for a keypoint in the names list."""
    kp_lower = keypoint.lower()
    idx_x, idx_y = None, None

    for i, name in enumerate(names):
        name_lower = str(name).lower()
        if kp_lower in name_lower:
            if '_x' in name_lower or name_lower.endswith('x'):
                idx_x = i
            elif '_y' in name_lower or name_lower.endswith('y'):
                idx_y = i

    return idx_x, idx_y


def plot_xy_trajectory(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    title: str,
    is_removed: bool = False,
    removal_reason: str = "",
):
    """
    Plot X-Y trajectory with time coloring.

    Parameters:
        ax: matplotlib axis
        x, y: position arrays
        t: time array (for coloring)
        title: subplot title
        is_removed: whether trial was removed/gated out
        removal_reason: string reason for removal
    """
    valid = np.isfinite(x) & np.isfinite(y)

    if np.sum(valid) < 2:
        empty_text = "No Data\n[REMOVED]" if is_removed else "No Data"
        ax.text(
            0.5, 0.5,
            empty_text,
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=7,
            color="#c62828" if is_removed else "#777777",
            fontweight="bold" if is_removed else "normal",
        )
        ax.set_title(
            title,
            fontsize=6.5,
            color="#c62828" if is_removed else "#111111",
            fontweight="bold" if is_removed else "normal",
            pad=3,
        )
        ax.tick_params(labelsize=5)
        if is_removed:
            for spine in ax.spines.values():
                spine.set_edgecolor("#d32f2f")
                spine.set_linewidth(2.8)
            ax.set_facecolor("#fff0f0")
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor("#cccccc")
                spine.set_linewidth(0.8)
            ax.set_facecolor("white")
        return

    # Create line segments for color mapping
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by time
    norm = Normalize(vmin=t.min(), vmax=t.max())
    alpha = 0.75 if is_removed else 0.85
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=1.5, alpha=alpha)
    lc.set_array(t[:-1])
    ax.add_collection(lc)

    # Mark start (green) and end (red) using first and last valid points
    valid_indices = np.where(valid)[0]
    ax.scatter(x[valid_indices[0]], y[valid_indices[0]], c='green', s=30, zorder=5, label='Start')
    ax.scatter(x[valid_indices[-1]], y[valid_indices[-1]], c='red', s=30, zorder=5, label='End')

    # Mark t=0 if in range
    t0_idx = np.argmin(np.abs(t))
    if 0 < t0_idx < len(x) - 1 and np.isfinite(x[t0_idx]) and np.isfinite(y[t0_idx]):
        ax.scatter(x[t0_idx], y[t0_idx], c='blue', s=40, marker='x', zorder=6, linewidths=2)

    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(
        title,
        fontsize=6.5,
        color="#c62828" if is_removed else "#111111",
        fontweight="bold" if is_removed else "normal",
        pad=3,
    )
    ax.tick_params(labelsize=5)

    # Apply red highlight styling around removed trials
    if is_removed:
        for spine in ax.spines.values():
            spine.set_edgecolor("#d32f2f")
            spine.set_linewidth(2.8)
        ax.set_facecolor("#fff0f0")
    else:
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.8)
        ax.set_facecolor("white")


def find_all_peristim_files() -> list:
    """Find all PeriStim .npz files."""
    files = []

    if PERI_ROOT is None or not Path(PERI_ROOT).exists():
        return files

    # Stim and control reaches
    for reach_type in ['stim_reaches', 'control_reaches']:
        for target in ['target_A', 'target_B']:
            target_dir = PERI_ROOT / reach_type / target
            if target_dir.exists():
                cond_type = 'stim' if 'stim' in reach_type else 'control'
                for f in sorted(target_dir.glob("*.npz")):
                    files.append((f, cond_type, target))

    # Continuous stim
    continuous_dir = PERI_ROOT / "continuous_stim"
    if continuous_dir.exists():
        for f in sorted(continuous_dir.glob("*.npz")):
            files.append((f, 'continuous_stim', 'continuous_stim'))

    # At rest
    at_rest_dir = PERI_ROOT / "at_rest"
    if at_rest_dir.exists():
        for f in sorted(at_rest_dir.glob("*.npz")):
            files.append((f, 'at_rest', 'at_rest'))

    return files


def process_single_file(
    npz_path: Path,
    condition_type: str,
    target_label: str,
    df_manual_rules: pd.DataFrame,
    active_process_only: list[int] | None = None,
):
    """Process one file and generate X-Y trajectory grid with removed trials highlighted in red."""

    try:
        data = np.load(npz_path, allow_pickle=True)

        # Get metadata
        br_idx = int(data["br_idx"]) if "br_idx" in data else 0
        intan_filename = str(data["sess"]) if "sess" in data else "Unknown"

        # Check process_only
        if active_process_only is not None and br_idx not in active_process_only:
            return

        raw_trial_indices = data["raw_trial_indices"] if "raw_trial_indices" in data.files else None

        print(f"[IKT] Processing BR {br_idx}, {intan_filename}, {condition_type}, {target_label}:")

        # Get time axis
        t = data["beh_rel_t"] if "beh_rel_t" in data.files else None
        if t is None or not hasattr(t, "size") or t.size < 2 or not np.isfinite(t).any():
            t = np.linspace(-800.0, 600.0, 141)

        # Get segments for selected camera
        cam_key = f"beh_{CAMERA}_segs"
        names_key = f"beh_{CAMERA}_names"

        has_cam = cam_key in data.files
        segs = data[cam_key] if has_cam else None
        names = data[names_key].tolist() if names_key in data.files else []

        # Find X and Y indices for keypoint
        idx_x, idx_y = find_xy_indices(names, KEYPOINT) if names else (None, None)
        if not has_cam:
            print(f"[IKT] Note: No {CAMERA} data found in {npz_path.name}")
        elif idx_x is None or idx_y is None:
            print(f"[IKT] Note: Could not find {KEYPOINT} X/Y in {CAMERA} names: {names}")

        # Query manual removal rules for this session and BR
        manual_rules = get_manual_removed_trials_for_session(df_manual_rules, intan_filename, br_idx)

        all_trials = []
        raw_in_peristim = set()

        # 1. Collect kept trials from PeriStim
        if segs is not None and segs.ndim == 3 and segs.shape[0] > 0 and idx_x is not None and idx_y is not None:
            for i in range(segs.shape[0]):
                raw_idx = int(raw_trial_indices[i]) if (raw_trial_indices is not None and i < len(raw_trial_indices)) else i
                raw_in_peristim.add(raw_idx)

                # Check if marked in manual_trial_remove.csv (even if not yet re-extracted)
                is_removed = raw_idx in manual_rules
                reason = manual_rules.get(raw_idx, "")

                all_trials.append({
                    "raw_idx": raw_idx,
                    "x": segs[i, idx_x, :],
                    "y": segs[i, idx_y, :],
                    "is_removed": is_removed,
                    "reason": reason,
                    "source": "peristim",
                })
        elif raw_trial_indices is not None and len(raw_trial_indices) > 0:
            # PeriStim has trial indices, but no behavior segments (e.g. no tracking / no behavior time axis)
            for r_idx in raw_trial_indices:
                raw_idx = int(r_idx)
                raw_in_peristim.add(raw_idx)
                reason = manual_rules.get(raw_idx, "no behavior tracking")
                all_trials.append({
                    "raw_idx": raw_idx,
                    "x": np.full(len(t), np.nan),
                    "y": np.full(len(t), np.nan),
                    "is_removed": True,
                    "reason": reason,
                    "source": "peristim_no_beh",
                })
        elif "event_ms" in data.files and len(data["event_ms"]) > 0:
            # Fallback if raw_trial_indices is not present but event_ms is
            for i in range(len(data["event_ms"])):
                raw_idx = i
                raw_in_peristim.add(raw_idx)
                reason = manual_rules.get(raw_idx, "no behavior tracking")
                all_trials.append({
                    "raw_idx": raw_idx,
                    "x": np.full(len(t), np.nan),
                    "y": np.full(len(t), np.nan),
                    "is_removed": True,
                    "reason": reason,
                    "source": "peristim_no_beh",
                })

        # 2. Check for removed trials already stored in PeriStim
        cam_rem_key = f"beh_{CAMERA}_segs_removed"
        if cam_rem_key in data.files and "raw_trial_indices_removed" in data.files:
            segs_rem = data[cam_rem_key]
            raw_rem = data["raw_trial_indices_removed"]
            reasons_rem = data["removal_reasons"] if "removal_reasons" in data.files else None
            if segs_rem.ndim == 3 and idx_x is not None and idx_y is not None:
                for j in range(segs_rem.shape[0]):
                    r_idx = int(raw_rem[j]) if j < len(raw_rem) else j
                    if r_idx not in raw_in_peristim:
                        raw_in_peristim.add(r_idx)
                        r_reason = str(reasons_rem[j]) if (reasons_rem is not None and j < len(reasons_rem)) else manual_rules.get(r_idx, "removed")
                        all_trials.append({
                            "raw_idx": r_idx,
                            "x": segs_rem[j, idx_x, :],
                            "y": segs_rem[j, idx_y, :],
                            "is_removed": True,
                            "reason": r_reason,
                            "source": "peristim_removed",
                        })

        # 3. Retrieve any remaining missing/removed trials from Aligned checkpoint
        target_letter = None
        if "target_A" in target_label or target_label.upper() == "A":
            target_letter = "A"
        elif "target_B" in target_label or target_label.upper() == "B":
            target_letter = "B"

        missing_from_aligned = find_and_extract_missing_trials(
            intan_filename=intan_filename,
            br_idx=br_idx,
            existing_raw_indices=raw_in_peristim,
            manual_rules=manual_rules,
            target_letter=target_letter,
            t_peri=t,
            keypoint=KEYPOINT,
            camera=CAMERA,
            condition_type=condition_type,
        )
        for tr in missing_from_aligned:
            if tr["raw_idx"] not in raw_in_peristim:
                raw_in_peristim.add(tr["raw_idx"])
                all_trials.append(tr)

        # 4. If all_trials is still empty, populate from manual rules or placeholder
        if not all_trials:
            if manual_rules:
                for r_idx in sorted(manual_rules.keys()):
                    all_trials.append({
                        "raw_idx": r_idx,
                        "x": np.full(len(t), np.nan),
                        "y": np.full(len(t), np.nan),
                        "is_removed": True,
                        "reason": manual_rules[r_idx],
                        "source": "manual_rules",
                    })
            else:
                all_trials.append({
                    "raw_idx": 0,
                    "x": np.full(len(t), np.nan),
                    "y": np.full(len(t), np.nan),
                    "is_removed": True,
                    "reason": "no valid trials found",
                    "source": "placeholder",
                })

        # Sort all trials by raw stim index for consistent chronological ordering
        all_trials.sort(key=lambda item: item["raw_idx"])
        n_trials = len(all_trials)

        # Count how many trials have valid kinematics data AND are not removed
        valid_trials_count = sum(
            1 for tr in all_trials
            if (not tr["is_removed"]) and np.sum(np.isfinite(tr["x"]) & np.isfinite(tr["y"])) >= 2
        )

        if valid_trials_count == 0:
            print(f"[IKT] No valid trials found ({n_trials} trials total), plotting all as removed (red)")
            for tr in all_trials:
                tr["is_removed"] = True
                if not tr["reason"]:
                    tr["reason"] = "no valid kinematics"
        else:
            # For any trials with insufficient data points, mark them removed
            for tr in all_trials:
                if not tr["is_removed"] and np.sum(np.isfinite(tr["x"]) & np.isfinite(tr["y"])) < 2:
                    tr["is_removed"] = True
                    if not tr["reason"]:
                        tr["reason"] = "no data"

        # Count kept vs removed
        n_removed = sum(1 for tr in all_trials if tr["is_removed"])
        n_kept = n_trials - n_removed

        print(f"[IKT] Total trials: {n_trials} ({n_kept} kept, {n_removed} highlighted removed)")

        # Cap trials to prevent matplotlib / PIL image dimension overflow (max 65500 px)
        MAX_TRIALS_PER_FIG = 180  # 30 rows of 6 = 180 subplots (~78 inches tall = 11700 px at 150 DPI)
        if n_trials > MAX_TRIALS_PER_FIG:
            print(f"[IKT] Warning: {n_trials} trials exceeds maximum display limit ({MAX_TRIALS_PER_FIG}). Capping to first {MAX_TRIALS_PER_FIG} trials to avoid figure overflow.")
            all_trials = all_trials[:MAX_TRIALS_PER_FIG]
            n_trials = len(all_trials)

        fig = None
        try:
            # Create figure grid
            n_rows = math.ceil(n_trials / N_COLS)
            fig, axes = plt.subplots(
                n_rows, N_COLS,
                figsize=(FIG_WIDTH_PER_COL * N_COLS, FIG_HEIGHT_PER_ROW * n_rows),
                constrained_layout=True
            )

            # Ensure axes is 2D array
            if n_rows == 1:
                axes = np.array([axes])
            if axes.ndim == 1:
                axes = axes.reshape(1, -1)

            axes_flat = axes.flatten()

            # Plot each trial
            for i, tr in enumerate(all_trials):
                ax = axes_flat[i]
                x = tr["x"]
                y = tr["y"]
                raw_idx = tr["raw_idx"]
                is_rem = tr["is_removed"]
                rem_reason = tr["reason"]

                sess_display = intan_filename[:16] if len(intan_filename) >= 16 else intan_filename
                if is_rem:
                    rem_display = rem_reason if len(rem_reason) <= 28 else (rem_reason[:25] + "...")
                    reason_str = f"\n[REMOVED: {rem_display}]" if rem_display else "\n[REMOVED]"
                else:
                    reason_str = ""

                title = f"{sess_display}\nBR:{br_idx} | Trial:{i+1} (Raw:{raw_idx}){reason_str}"
                plot_xy_trajectory(ax, x, y, t, title, is_removed=is_rem, removal_reason=rem_reason)

            # Hide unused axes
            for i in range(n_trials, len(axes_flat)):
                axes_flat[i].axis('off')

            # Main overall title
            if "overall_title" in data.files:
                raw_title = data["overall_title"]
                overall_title = str(raw_title.item()) if raw_title.shape == () else str(raw_title)
            else:
                overall_title = npz_path.stem

            if n_kept == 0:
                status_info = f"n={n_trials} (NO VALID TRIALS - all {n_removed} removed)"
            elif n_removed > 0:
                status_info = f"n={n_trials} ({n_kept} kept, {n_removed} removed)"
            else:
                status_info = f"n={n_trials} (all kept)"

            fig.suptitle(
                f"{overall_title}\n{KEYPOINT.title()} Finger X-Y | {condition_type} / {target_label} | {status_info}\n"
                f"Session: {intan_filename} | BR: {br_idx}  —  Red box = Removed trials | Put Raw:# in manual_trial_remove.csv",
                fontsize=10
            )

            # Add colorbar for time
            t_min = float(np.nanmin(t)) if np.isfinite(t).any() else -800.0
            t_max = float(np.nanmax(t)) if np.isfinite(t).any() else 600.0
            if t_min >= t_max:
                t_min, t_max = -800.0, 600.0
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=t_min, vmax=t_max))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes_flat[:n_trials], shrink=0.6, pad=0.02, aspect=30)
            cbar.set_label('Time (ms)', fontsize=8)

            # Save output figure
            out_dir = FIG_ROOT / condition_type / target_label
            out_dir.mkdir(parents=True, exist_ok=True)

            file_name = f"BR_{br_idx:03d}_xy.jpg" if br_idx else f"{npz_path.stem}_xy.jpg"
            out_path = out_dir / file_name

            fig.savefig(out_path, dpi=150)
            print(f"[IKT] Saved {out_path}")
        finally:
            if fig is not None:
                plt.close(fig)

    except Exception as e:
        print(f"[IKT] Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    if os.environ.get("RCP_VELES_RUN") != "1":
        from RCP_analysis.python.functions.pipeline_hierarchy import check_and_confirm_dependencies
        data_root = f"{PARAMS.data_root}/{PARAMS.monkey}"
        if not check_and_confirm_dependencies(Path(__file__).name, PARAMS.session, data_root):
            print("[inspect_kinematics_trajectories] Aborted by user due to dependency discrepancy.")
            return

    # Check for active process_only from environment or params
    active_process_only = None
    env_po = os.environ.get("RCP_PROCESS_ONLY")
    if env_po:
        try:
            parsed = json.loads(env_po)
            if isinstance(parsed, list) and len(parsed) > 0:
                active_process_only = [int(x) for x in parsed if str(x).lstrip("-").isdigit()]
        except Exception:
            pass
    if active_process_only is None and PARAMS and hasattr(PARAMS, "process_only") and PARAMS.process_only:
        try:
            active_process_only = [int(x) for x in PARAMS.process_only if str(x).lstrip("-").isdigit()]
        except Exception:
            pass

    print(f"\n[IKT] Session: {PARAMS.session}, Camera: {CAMERA}, Keypoint: {KEYPOINT}")
    if active_process_only:
        print(f"[IKT] Process Only BR conditions: {active_process_only}")

    df_manual_rules = load_manual_remove_rules()
    if not df_manual_rules.empty:
        print(f"[IKT] Loaded {len(df_manual_rules)} manual removal rules from {MANUAL_REMOVE_CSV.name}")

    all_files = find_all_peristim_files()

    if not all_files:
        print(f"[IKT] No .npz files found in {PERI_ROOT}")
        return

    print(f"\n[IKT] Found {len(all_files)} files\n")

    for npz_path, condition_type, target_label in all_files:
        process_single_file(
            npz_path,
            condition_type,
            target_label,
            df_manual_rules=df_manual_rules,
            active_process_only=active_process_only,
        )

    print(f"\n[IKT] Done!")


if __name__ == "__main__":
    main()