"""
This script plots the firing rate plots based on Gaussian smoothing parameters
Steps:
    1. Plot aligned baseline traces
    2. Plot aligned condition traces
    3. Plot aligned at rest traces
    4. These plots include (Median, variance, mean traces, median count traces, and chosen / best DLC coordinates)
    4, Plot first 4 trials
"""
from types import SimpleNamespace
import numpy as np
import matplotlib

from scipy.io import loadmat
from probeinterface import Probe

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

PROCESS_ONLY = PARAMS.preprocessing.get("process_only")
Z_SCORE_FR   = PARAMS.preprocessing.get("z_score_firing_rate", False)
NPRW_BLANK_BEFORE_MS = float(PARAMS.NPRW_rate_est.get("remove_ms_before", 20.0))
NPRW_BLANK_AFTER_MS  = float(PARAMS.NPRW_rate_est.get("remove_tail_ms_after", 20.0))

# Local plotting settings
matplotlib.rcParams["svg.fonttype"] = "none"

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
NORMALIZE_FIRST_MS = 150.0

# Neural heatmap vmin/vmax for median
VMIN_NPRW, VMAX_NPRW = -25, 200
VMIN_UA,   VMAX_UA   = -50, 150

# Neural heatmap vmin/vmax for variance
VMIN_NPRW_VAR, VMAX_NPRW_VAR = 0.0, 40000.0
VMIN_UA_VAR,   VMAX_UA_VAR   = 0.0, 10000.0
VMAX_SMA_VAR                 = 5000.0

VMIN_NPRW_COUNTS, VMAX_NPRW_COUNTS = 0.0, 10.0
VMIN_UA_COUNTS,   VMAX_UA_COUNTS   = 0.0, 10.0

# Override color ranges for z-scored firing rates
if Z_SCORE_FR:
    VMIN_NPRW, VMAX_NPRW = -3.0, 3.0
    VMIN_UA,   VMAX_UA   = -3.0, 3.0

COLORMAP = "RdBu_r"

# Kinematics
KINEMATICS_YLIM = (-4, 4)    # fixed y-limits for all figures
KINEMATICS_YLIM_NORMDIST = (-0.1, 1.1)

KIN_KEYPOINT_INCLUDE = ("middle",) #, "wrist", "ALL")

NORMALIZED_DISTANCE = True # False if we want individual x and y positions plotted instead
NORMALIZED_DISTANCE_KEYPOINT = "middle"
NORMALIZED_DISTANCE_REF_TIME_MS = -600.0 # -600 used in plot_plateau_analysis.py

# "max_abs"   : divide by max(abs(distance)) over time/trial.
# "final_abs" : divide by abs(distance at final valid time).
# "none"      : no scaling; plots distance-from-reference in original z/pixel units.
NORMALIZED_DISTANCE_MODE = "max_abs"

PLOT_POSITION = True
PLOT_VELOCITY = False

# Layout knobs (passed into stacked_heatmaps_plus_behv)
BEH_RATIO = 0.6               # height ratio for behavior rows (position/velocity); adjust as needed
CH_RATIO_PER_ROW = 0.015      # height ratio per neural channel row (heatmaps)
MIN_HEATMAP_RATIO = 0.6       # minimum ratio so tiny arrays don't vanish
UA_COMPACT_FACTOR = 0.95      # < 1.0 shrinks UA panel heights
NPRW_SCALE        = 0.6       # < 1.0 shrinks Intan height (e.g., 0.6 = 60% of previous)
GAP_BEH_NPRW      = 0.25      # height "ratio" for a spacer row between behavior and Intan
FIG_WIDTH_IN         = 8.0    # overall width (inches)
HEIGHT_PER_RATIO_IN  = 4.0    # height per unit of `ratios` sum
PROBE_GAP_RATIO      = 0.15
PROBE_WIDTH_RATIO    = 0.35

SKIP_EXISTING        = False   # If True, skip sessions where the first plot already exists

# ---------------------------------------------------------------------
# ROOTS / PARAMS
# ---------------------------------------------------------------------
# Base paths from config_loading

# Figures
FIG_ROOT   = OUT_BASE / "figures/shaded_BT_svg"; FIG_ROOT.mkdir(parents=True, exist_ok=True)
FIG = SimpleNamespace(
    peri_posvel_median  = FIG_ROOT / "median_fr_plots",
    peri_posvel_meanMWT = FIG_ROOT / "mean_MWT_plots",
)
FIG.peri_posvel_median.mkdir(parents=True, exist_ok=True)
FIG.peri_posvel_meanMWT.mkdir(parents=True, exist_ok=True)
FIG.peri_var_median    = FIG_ROOT / "variance_fr_plots"; FIG.peri_var_median.mkdir(parents=True, exist_ok=True)
FIG.peri_var_meanMWT   = FIG_ROOT / "variance_MWT_plots"; FIG.peri_var_meanMWT.mkdir(parents=True, exist_ok=True)
FIG.peri_counts_meanMWT = FIG_ROOT / "median_count_MWT_plots"; FIG.peri_counts_meanMWT.mkdir(parents=True, exist_ok=True)
FIG.peri_single_trials = FIG_ROOT / "single_trial_fr_plots"; FIG.peri_single_trials.mkdir(parents=True, exist_ok=True)

# Peri-stim checkpoints
PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"

# NPC aux / stim
NPRW_AUX_DATA = OUT_BASE / "aux_data" / "NPRW"

# NPRW mapping / geometry
GEOM_PATH = (
    Path(PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
    else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None)
    else rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
)

# Tuple order for keypoints
KEYPOINTS_ORDER = tuple(PARAMS.kinematics.get("keypoints"))

def _compute_target_means_for_cam(
    cam_pos_segs: np.ndarray,
    cam_names: list[str],
    cam_rel_t: np.ndarray,          # (T_pos,)
    ts_state_segs: np.ndarray,      # (n_trials, T_state)
    ts_state_rel_t: np.ndarray,     # (T_state,)
    target_suffix: str,
):
    """
    Uses ts_state == 1 as the 'on' state and resamples ts_state onto cam_rel_t
    using nearest-neighbor in time.

    Returns
    -------
    idx_mask : (n_kps,) bool
    mean_xy  : (2,) global mean [x, y] when ts_state==1
    mean_xy_per_trial : (n_trials, 2) per-trial mean [x, y] when ts_state==1
    """
    if cam_pos_segs is None or np.size(cam_pos_segs) == 0 or cam_names is None or len(np.atleast_1d(cam_names)) == 0:
        return None, None, None
    if ts_state_segs is None or ts_state_segs.size == 0:
        return None, None, None

    cam_rel_t = np.asarray(cam_rel_t, float).ravel()
    ts_state_rel_t = np.asarray(ts_state_rel_t, float).ravel()
    if cam_rel_t.size == 0 or ts_state_rel_t.size == 0:
        return None, None, None

    cam_names_arr = np.asarray(cam_names)
    idx_mask = np.isin(cam_names_arr, [f"{target_suffix}_x", f"{target_suffix}_y"])
    if not idx_mask.any():
        return idx_mask, None, None

    # (n_trials, 2, T_pos)
    target_pos = np.asarray(cam_pos_segs, float)[:, idx_mask, :]
    n_trials, _, T_pos = target_pos.shape

    ts_state = np.asarray(ts_state_segs)
    if ts_state.ndim != 2:
        raise ValueError(f"ts_state_segs must be 2D (n_trials, T_state), got {ts_state.shape}")
    if ts_state.shape[0] != n_trials:
        # Instead of raising error, truncate to common length
        print(f"[warn] Trial mismatch: target_pos has {n_trials} trials but ts_state has {ts_state.shape[0]}. Truncating to min.")
        n_min = min(n_trials, ts_state.shape[0])
        ts_state = ts_state[:n_min]
        target_pos = target_pos[:n_min]
        n_trials = n_min

    # On-state on ts grid
    state_on_ts = (ts_state == 1)  # (n_trials, T_state)

    # If grids already match, no resampling needed
    if (T_pos == ts_state_rel_t.size) and np.allclose(cam_rel_t, ts_state_rel_t, atol=1e-6, rtol=0):
        state_on_cam = state_on_ts
    else:
        # nearest-neighbor mapping from cam_rel_t -> ts_state_rel_t
        idx = np.searchsorted(ts_state_rel_t, cam_rel_t, side="left")
        idx = np.clip(idx, 0, ts_state_rel_t.size - 1)

        left = np.clip(idx - 1, 0, ts_state_rel_t.size - 1)
        choose_left = (
            np.abs(cam_rel_t - ts_state_rel_t[left])
            <= np.abs(ts_state_rel_t[idx] - cam_rel_t)
        )
        idx = np.where(choose_left, left, idx)

        state_on_cam = state_on_ts[:, idx]  # (n_trials, T_pos)

    # Broadcast mask and apply
    mask = state_on_cam[:, None, :]                 # (n_trials, 1, T_pos)
    pos_masked = np.where(mask, target_pos, np.nan) # (n_trials, 2, T_pos)

    mean_xy = np.nanmean(pos_masked, axis=(0, 2))      # (2,)
    mean_xy_per_trial = np.nanmean(pos_masked, axis=2) # (n_trials, 2)

    return idx_mask, mean_xy, mean_xy_per_trial


def _compute_mean_and_sd(segs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    segs : (n_trials, K, T)
    Returns
    -------
    mean : (K, T)
    sd  : (K, T)
    """
    if segs is None or segs.size == 0:
        return np.zeros((0, 0), float), np.zeros((0, 0), float)

    segs = np.asarray(segs, float)
    mean = np.nanmean(segs, axis=0)          # (K, T)
    sd   = np.nanstd(segs, axis=0)
    cnt  = np.sum(np.isfinite(segs), axis=0) # (K, T), per-trace per-time
    with np.errstate(divide="ignore", invalid="ignore"):
        std = sd / np.sqrt(cnt)
        std[cnt == 0] = np.nan
    mean[cnt == 0] = np.nan
    return mean, sd


def _select_kinematics_for_plot(
    beh_labels_display,
    beh_cam0_pos=None,
    beh_cam1_pos=None,
    beh_cam0_vel=None,
    beh_cam1_vel=None,
    beh_cam0_pos_stds=None,
    beh_cam1_pos_stds=None,
    beh_cam0_vel_stds=None,
    beh_cam1_vel_stds=None,
    keypoint_include=KIN_KEYPOINT_INCLUDE,
    plot_position=PLOT_POSITION,
    plot_velocity=PLOT_VELOCITY,
):
    idx_subset = rcp.get_subset_indices(
        beh_labels_display,
        keys=keypoint_include,
    )

    if not idx_subset:
        return {
            "beh_labels": [],
            "beh_cam0_pos": None,
            "beh_cam1_pos": None,
            "beh_cam0_vel": None,
            "beh_cam1_vel": None,
            "beh_cam0_pos_stds": None,
            "beh_cam1_pos_stds": None,
            "beh_cam0_vel_stds": None,
            "beh_cam1_vel_stds": None,
        }

    beh_labels_subset = [beh_labels_display[i] for i in idx_subset]

    def _slice(arr):
        if arr is None:
            return None

        arr = np.asarray(arr)

        if arr.size == 0:
            return None

        # Case 1: mean/median traces, shape: (K, T)
        if arr.ndim == 2:
            valid = [i for i in idx_subset if 0 <= i < arr.shape[0]]
            if not valid:
                return None
            return arr[valid, :]

        # Case 2: trial traces, shape: (n_trials, K, T)
        if arr.ndim == 3:
            valid = [i for i in idx_subset if 0 <= i < arr.shape[1]]
            if not valid:
                return None
            return arr[:, valid, :]

        raise ValueError(f"Unexpected behavior array shape: {arr.shape}")

    return {
        "beh_labels": beh_labels_subset,

        "beh_cam0_pos": _slice(beh_cam0_pos) if plot_position else None,
        "beh_cam1_pos": _slice(beh_cam1_pos) if plot_position else None,
        "beh_cam0_vel": _slice(beh_cam0_vel) if plot_velocity else None,
        "beh_cam1_vel": _slice(beh_cam1_vel) if plot_velocity else None,

        "beh_cam0_pos_stds": _slice(beh_cam0_pos_stds) if plot_position else None,
        "beh_cam1_pos_stds": _slice(beh_cam1_pos_stds) if plot_position else None,
        "beh_cam0_vel_stds": _slice(beh_cam0_vel_stds) if plot_velocity else None,
        "beh_cam1_vel_stds": _slice(beh_cam1_vel_stds) if plot_velocity else None,
    }



# ---------------------------------------------------------------------
# MAIN PERI-STIM PLOTTING
# ---------------------------------------------------------------------
def main():
    # ---- load probe geometry once ----
    mat_probe = loadmat(Path(GEOM_PATH))
    nprw_geom = {
        "x": mat_probe["xcoords"].ravel(),
        "y": mat_probe["ycoords"].ravel(),
    }
    assert nprw_geom["x"].size == nprw_geom["y"].size

    if "chanMap0ind" in mat_probe:
        dev_idx = mat_probe["chanMap0ind"].ravel()
    else:
        raise ValueError("No 0-based chanMap0ind in .mat geometry file.")

    if dev_idx.size != nprw_geom["x"].size:
        raise ValueError("device_index_0based length != #contacts")

    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(
        positions=np.c_[nprw_geom["x"], nprw_geom["y"]],
        shapes="square",
        shape_params={"width": 12.0},
    )
    nprw_probe.set_device_channel_indices(dev_idx)
    locs = nprw_probe.contact_positions.astype(float)

    # -----------------------------------------------------------------
    # Search all subdirectories in PERI_ROOT
    # -----------------------------------------------------------------
    if not PERI_ROOT.exists():
        print(f"[warn] PERI_ROOT does not exist: {PERI_ROOT}")
        return

    files = sorted(PERI_ROOT.rglob("*.npz"))

    if not files:
        print(f"[warn] No .npz files found in {PERI_ROOT}")
        return

    for k, peri_stim_npz_loc in enumerate(files):
        # if k <= 2:
        #     continue
        if not peri_stim_npz_loc.exists():
            continue

        target_label = str(peri_stim_npz_loc.parent.relative_to(PERI_ROOT)).replace("\\", "/")
        if target_label == ".":
            target_label = peri_stim_npz_loc.parent.name
        
        # --- PRE-LOAD SKIP CHECK ---
        if SKIP_EXISTING:
            out_dir_check = FIG.peri_posvel_median / target_label
            
            # Derive expected output filename from input filename
            # Pattern: peristim__{session}__BR_{br_idx}_...
            fname = peri_stim_npz_loc.name
            check_file = None
            
            if fname.startswith('baseline'):
                check_file = f"{fname}__median_ALL.svg"
            else:
                # Try to extract BR index
                import re
                match = re.search(r"BR_(\d+)", fname)
                if match:
                    br_val = int(match.group(1))
                    check_file = f"Cond_{br_val:03d}__median_ALL.svg"
            
            if check_file:
                check_path = out_dir_check / check_file
                if check_path.exists():
                    print(f"  [Skip] {check_file} exists (pre-load check).")
                    continue

        print(f"Processing {peri_stim_npz_loc.name} ({target_label})...")

        peri_stim_npz = np.load(peri_stim_npz_loc, allow_pickle=True)
        sess         = str(peri_stim_npz["sess"])
        br_idx       = int(peri_stim_npz["br_idx"])

        meta_raw = peri_stim_npz["meta"].item() if "meta" in peri_stim_npz.files else {}

        stim_dur_ms = (
            float(meta_raw.get("recording_stim_dur", 0.0))
            if isinstance(meta_raw, dict)
            else 0.0
        )

        is_control = "control_reaches" in target_label.lower()

        nprw_blank_ms = None if is_control else (
            -NPRW_BLANK_BEFORE_MS,
            stim_dur_ms + NPRW_BLANK_AFTER_MS,
        )

        if PROCESS_ONLY and br_idx not in PROCESS_ONLY:
            continue
        overall_title_raw = peri_stim_npz["overall_title"]
        if overall_title_raw.shape == ():
            overall_title = str(overall_title_raw.item())
        else:
            overall_title = str(overall_title_raw)

        # --- behavior ---
        beh_rel_t           = peri_stim_npz["beh_rel_t"]
        beh_cam0_pos_med    = peri_stim_npz["beh_cam0_pos_med"]
        beh_cam1_pos_med    = peri_stim_npz["beh_cam1_pos_med"]
        beh_cam0_vel_med    = peri_stim_npz["beh_cam0_vel_med"]
        beh_cam1_vel_med    = peri_stim_npz["beh_cam1_vel_med"]

        cam0_pos_segs = peri_stim_npz["beh_cam0_segs"]
        cam1_pos_segs = peri_stim_npz["beh_cam1_segs"]
        cam0_vel_segs = peri_stim_npz["beh_cam0_vel_segs"]
        cam1_vel_segs = peri_stim_npz["beh_cam1_vel_segs"]

        raw_c0       = peri_stim_npz["beh_cam0_names"] if "beh_cam0_names" in peri_stim_npz.files else []
        raw_c1       = peri_stim_npz["beh_cam1_names"] if "beh_cam1_names" in peri_stim_npz.files else []
        cam0_names   = raw_c0.tolist() if isinstance(raw_c0, np.ndarray) else raw_c0
        cam1_names   = raw_c1.tolist() if isinstance(raw_c1, np.ndarray) else raw_c1

        ts_state_segs = peri_stim_npz["ts_state_segs"]  # (n_trials, T_state)
        ts_state_rel_t = peri_stim_npz["ts_state_rel_t"]  # (n_trials, T_state)

        # Decide which target to use based on folder name
        parent_name  = peri_stim_npz_loc.parent.name.lower()
        if parent_name == "target_a":
            target_suffix = "ta"
        elif parent_name == "target_b":
            target_suffix = "tb"
        else:
            target_suffix = ""
            
        _, cam0_mean_xy, _ = _compute_target_means_for_cam(
            cam0_pos_segs, cam0_names, beh_rel_t, ts_state_segs, ts_state_rel_t, target_suffix
        )
        _, cam1_mean_xy, _ = _compute_target_means_for_cam(
            cam1_pos_segs, cam1_names, beh_rel_t, ts_state_segs, ts_state_rel_t, target_suffix
        )

        target_pos_cam0 = cam0_mean_xy if cam0_mean_xy is not None else None
        target_pos_cam1 = cam1_mean_xy if cam1_mean_xy is not None else None
        
        # --- neural ---
        NPRW_rates_zeroed = peri_stim_npz["NPRW_rates_zeroed"]
        NPRW_med          = peri_stim_npz["NPRW_med"]
        NPRW_med_counts   = peri_stim_npz["NPRW_med_counts"]
        NPRW_var          = peri_stim_npz["NPRW_var"]
        NPRW_rel_t        = peri_stim_npz["NPRW_rel_t"]
        NPRW_width_ms     = peri_stim_npz["NPRW_width_ms"]
        
        n_events = (
            int(peri_stim_npz["n_trials"])
            if "n_trials" in peri_stim_npz.files
            else int(np.asarray(peri_stim_npz["event_ms"]).size)
            if "event_ms" in peri_stim_npz.files
            else int(NPRW_rates_zeroed.shape[0])
            if NPRW_rates_zeroed.ndim == 3
            else 0
        )

        UA_rates_zeroed = (
            peri_stim_npz["UA_rates_zeroed"]
            if "UA_rates_zeroed" in peri_stim_npz.files
            else np.zeros((0, 0, 0), float)
        )
        UA_med        = peri_stim_npz["UA_med"]
        UA_var        = peri_stim_npz["UA_var"]
        UA_med_counts = peri_stim_npz["UA_med_counts"]
        UA_rel_t      = peri_stim_npz["UA_rel_t"]
        UA_width_ms   = peri_stim_npz["UA_width_ms"]
        ua_ids_1based = peri_stim_npz["ua_ids_1based"] if "ua_ids_1based" in peri_stim_npz.files else None


        # --- Optional z-score normalization ---
        if Z_SCORE_FR:
            if NPRW_rates_zeroed.ndim == 3 and NPRW_rates_zeroed.shape[0] > 0:
                mu  = np.nanmean(NPRW_rates_zeroed, axis=2, keepdims=True)
                sig = np.clip(np.nanstd(NPRW_rates_zeroed, axis=2, keepdims=True), 1e-6, None)
                NPRW_med = np.nanmean((NPRW_rates_zeroed - mu) / sig, axis=0)
            if UA_rates_zeroed.ndim == 3 and UA_rates_zeroed.shape[0] > 0:
                mu  = np.nanmean(UA_rates_zeroed, axis=2, keepdims=True)
                sig = np.clip(np.nanstd(UA_rates_zeroed, axis=2, keepdims=True), 1e-6, None)
                UA_med = np.nanmean((UA_rates_zeroed - mu) / sig, axis=0)
            
        # -----------------------------------------------------------------
        # Labels
        # -----------------------------------------------------------------
        if cam0_names:
            beh_labels_raw = [rcp.strip_cam_prefix(n) for n in cam0_names]
        elif cam1_names:
            beh_labels_raw = [rcp.strip_cam_prefix(n) for n in cam1_names]
        else:
            beh_labels_raw = []

        beh_labels_display = rcp.simple_beh_labels(beh_labels_raw, KEYPOINTS_ORDER)
        beh_time_for_both = beh_rel_t

        if NORMALIZED_DISTANCE:
            cam0_dist_trials = rcp.compute_normalized_distance_from_xy_3d(
                cam0_pos_segs,
                beh_labels_display,
                beh_rel_t,
                keypoint=NORMALIZED_DISTANCE_KEYPOINT,
                ref_time_ms=NORMALIZED_DISTANCE_REF_TIME_MS,
                mode=NORMALIZED_DISTANCE_MODE,
            )

            cam1_dist_trials = rcp.compute_normalized_distance_from_xy_3d(
                cam1_pos_segs,
                beh_labels_display,
                beh_rel_t,
                keypoint=NORMALIZED_DISTANCE_KEYPOINT,
                ref_time_ms=NORMALIZED_DISTANCE_REF_TIME_MS,
                mode=NORMALIZED_DISTANCE_MODE,
            )

            beh_cam0_pos_for_plot = rcp.median_from_trial_traces(cam0_dist_trials)
            beh_cam1_pos_for_plot = rcp.median_from_trial_traces(cam1_dist_trials)


            kin_med_plot = _select_kinematics_for_plot(
                beh_labels_display=[""],
                beh_cam0_pos=beh_cam0_pos_for_plot,
                beh_cam1_pos=beh_cam1_pos_for_plot,
                beh_cam0_vel=None,
                beh_cam1_vel=None,
                keypoint_include="ALL",
            )

            # Target x/y coordinates do not make sense on a normalized-distance axis.
            target_pos_cam0_for_plot = None
            target_pos_cam1_for_plot = None

        else:
            cam0_dist_trials = None
            cam1_dist_trials = None

            kin_med_plot = _select_kinematics_for_plot(
                beh_labels_display=beh_labels_display,
                beh_cam0_pos=beh_cam0_pos_med,
                beh_cam1_pos=beh_cam1_pos_med,
                beh_cam0_vel=beh_cam0_vel_med,
                beh_cam1_vel=beh_cam1_vel_med,
            )

            target_pos_cam0_for_plot = target_pos_cam0
            target_pos_cam1_for_plot = target_pos_cam1

        beh_ylim_for_plot = KINEMATICS_YLIM_NORMDIST if NORMALIZED_DISTANCE else KINEMATICS_YLIM

        beh_pos_ylabel_for_plot = (
            "Normalized distance"
            if NORMALIZED_DISTANCE
            else "Position Δ (z)"
        )

        # Titles
        base_kin_title = f"Kinematics / n={n_events} events"
        neural_type = "z-scored" if Z_SCORE_FR else "median"
        base_neural_title = f"Neural Activity ({neural_type} Δ) / Referenced to first {int(NORMALIZE_FIRST_MS)} ms)"
        full_overall_title = f"{overall_title} {target_label.replace('_', ' ').replace('/', ' - ')}"
        cb_label = "z-score (std)" if Z_SCORE_FR else "Δ FR (Hz)"

        # ---- stim site detection (probe inset) ---- TODO this can be extracted from active_channels
        stim_npz = NPRW_AUX_DATA / f"{sess}_Intan_streams" / "stim_stream.npz"
        stim_locs = None
        if stim_npz.exists():
            try:
                stim_locs = rcp.detect_stim_channels_from_npz(
                    stim_npz, eps=1e-12, min_edges=1
                )
            except Exception as e:
                print(f"[warn] stim-site detection failed for {sess}, Condition {br_idx}: {e}")

        # -----------------------------------------------------------------
        # FIGURE 1: ALL MEDIAN TRACES (no shading, all keypoints)
        # -----------------------------------------------------------------
        out_dir_1_parent = FIG.peri_posvel_median / target_label
        out_dir_1_parent.mkdir(parents=True, exist_ok=True)
        if peri_stim_npz_loc.name.startswith('baseline'):
            file_name = f"{peri_stim_npz_loc.name}"
        else:
            file_name = f"Cond_{br_idx:03d}"
        out_path_1 = out_dir_1_parent / f"{file_name}__median_ALL.svg"

        if SKIP_EXISTING and out_path_1.exists():
            print(f"  [Skip] {file_name} exists.")
            continue

        rcp.stacked_heatmaps_plus_behv(
            NPRW_med, UA_med,
            NPRW_rel_t if (NPRW_med.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_med.size   and UA_rel_t.size)   else None,
            NPRW_width_ms,
            UA_width_ms,
            out_path_1,
            base_kin_title,
            base_neural_title,
            beh_pos_ylabel=beh_pos_ylabel_for_plot,
            cmap=COLORMAP,
            cb_label_nprw=cb_label,
            cb_label_ua=cb_label,
            vmin_nprw=VMIN_NPRW, vmax_nprw=VMAX_NPRW,
            vmin_ua={
                "M1i": VMIN_UA,
                "M1s": VMIN_UA,
                "PMd": VMIN_UA,
                "SMA": VMIN_UA,
            },
            vmax_ua={
                "M1i": VMAX_UA,
                "M1s": VMAX_UA,
                "PMd": VMAX_UA,
                "SMA": VMAX_UA,
            },
            probe=nprw_probe,
            probe_locs=locs,
            stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="region_then_elec",
            beh_rel_time=beh_time_for_both,
            beh_cam0_pos=kin_med_plot["beh_cam0_pos"],
            beh_cam1_pos=kin_med_plot["beh_cam1_pos"],
            beh_cam0_vel=kin_med_plot["beh_cam0_vel"],
            beh_cam1_vel=kin_med_plot["beh_cam1_vel"],
            beh_cam0_pos_stds=None,  # no shading for median figure
            beh_cam1_pos_stds=None,
            beh_cam0_vel_stds=None,
            beh_cam1_vel_stds=None,
            target_pos_cam0=target_pos_cam0_for_plot,
            target_pos_cam1=target_pos_cam1_for_plot,
            beh_labels=kin_med_plot["beh_labels"],
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=full_overall_title,
            beh_ylim=beh_ylim_for_plot,
            beh_ratio=BEH_RATIO,
            ch_ratio_per_row=CH_RATIO_PER_ROW,
            min_heatmap_ratio=MIN_HEATMAP_RATIO,
            ua_compact_factor=UA_COMPACT_FACTOR,
            nprw_scale=NPRW_SCALE,
            gap_beh_nprw=GAP_BEH_NPRW,
            fig_width_in=FIG_WIDTH_IN,
            height_per_ratio_in=HEIGHT_PER_RATIO_IN,
            probe_gap_ratio=PROBE_GAP_RATIO,
            probe_width_ratio=PROBE_WIDTH_RATIO,
            nprw_blank_ms=nprw_blank_ms,
            stim_dur_ms=None if is_control else stim_dur_ms,
        )
        # -----------------------------------------------------------------
        # FIGURE 2: ALL VARIANCE TRACES (no shading, all keypoints)
        # -----------------------------------------------------------------
        out_dir_1b_parent = FIG.peri_var_median / target_label
        out_dir_1b_parent.mkdir(parents=True, exist_ok=True)
        out_path_1b = out_dir_1b_parent / f"{file_name}__var_ALL.svg"

        base_neural_var_title = f"Neural Variance (across {n_events} events)"

        rcp.stacked_heatmaps_plus_behv(
            NPRW_var, UA_var,
            NPRW_rel_t if (NPRW_var.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_var.size   and UA_rel_t.size)   else None,
            NPRW_width_ms,
            UA_width_ms,
            out_path_1b,
            base_kin_title,   # same behavior panel as median
            base_neural_var_title,
            beh_pos_ylabel=beh_pos_ylabel_for_plot,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW_VAR, vmax_nprw=VMAX_NPRW_VAR,
            vmin_ua={
                "M1i": VMIN_UA_VAR,
                "M1s": VMIN_UA_VAR,
                "PMd": VMIN_UA_VAR,
                "SMA": VMIN_UA_VAR,
            },
            vmax_ua={
                "M1i": VMAX_UA_VAR,
                "M1s": VMAX_UA_VAR,
                "PMd": VMAX_UA_VAR,
                "SMA": VMAX_SMA_VAR,
            },
            probe=nprw_probe,
            probe_locs=locs,
            stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="region_then_elec",
            beh_rel_time=beh_time_for_both,
            beh_cam0_pos=kin_med_plot["beh_cam0_pos"],
            beh_cam1_pos=kin_med_plot["beh_cam1_pos"],
            beh_cam0_vel=kin_med_plot["beh_cam0_vel"],
            beh_cam1_vel=kin_med_plot["beh_cam1_vel"],
            beh_cam0_pos_stds=None,
            beh_cam1_pos_stds=None,
            beh_cam0_vel_stds=None,
            beh_cam1_vel_stds=None,
            target_pos_cam0=target_pos_cam0_for_plot,
            target_pos_cam1=target_pos_cam1_for_plot,
            beh_labels=kin_med_plot["beh_labels"],
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=full_overall_title,
            beh_ylim=beh_ylim_for_plot,
            beh_ratio=BEH_RATIO,
            ch_ratio_per_row=CH_RATIO_PER_ROW,
            min_heatmap_ratio=MIN_HEATMAP_RATIO,
            ua_compact_factor=UA_COMPACT_FACTOR,
            nprw_scale=NPRW_SCALE,
            gap_beh_nprw=GAP_BEH_NPRW,
            fig_width_in=FIG_WIDTH_IN,
            height_per_ratio_in=HEIGHT_PER_RATIO_IN,
            probe_gap_ratio=PROBE_GAP_RATIO,
            probe_width_ratio=PROBE_WIDTH_RATIO,
            nprw_blank_ms=nprw_blank_ms,
            stim_dur_ms=None if is_control else stim_dur_ms,
        )

        # -----------------------------------------------------------------
        # FIGURE 3: MWT ONLY (Mean + shading)
        # -----------------------------------------------------------------
        out_dir_2_parent = FIG.peri_posvel_meanMWT / target_label
        out_dir_2_parent.mkdir(parents=True, exist_ok=True)
        out_path_2 = out_dir_2_parent / f"{file_name}__mean_MWT.svg"

        # Select MWT indices
        idx_subset = rcp.get_subset_indices(beh_labels_display, keys=KIN_KEYPOINT_INCLUDE)

        def _select_dims(segs, idxs):
            # Always return (subset, labels) – possibly (None, None)
            if segs is None or segs.size == 0 or not idxs:
                return None, None

            segs = np.asarray(segs, float)
            D = segs.shape[1]
            idxs = [i for i in idxs if 0 <= i < D]
            if not idxs:
                return None, None

            return segs[:, idxs, :], [beh_labels_display[i] for i in idxs]

        c0_pos_sel, labels_sel = _select_dims(cam0_pos_segs, idx_subset)
        c1_pos_sel, _          = _select_dims(cam1_pos_segs, idx_subset)
        c0_vel_sel, _          = _select_dims(cam0_vel_segs, idx_subset)
        c1_vel_sel, _          = _select_dims(cam1_vel_segs, idx_subset)

        # Compute mean + STD for pos, mean only for vel
        if c0_pos_sel is not None:
            c0_pos_mean, c0_pos_sd = _compute_mean_and_sd(c0_pos_sel)
        else:
            c0_pos_mean, c0_pos_sd = np.zeros((0, 0)), None

        if c1_pos_sel is not None:
            c1_pos_mean, c1_pos_sd = _compute_mean_and_sd(c1_pos_sel)
        else:
            c1_pos_mean, c1_pos_sd = np.zeros((0, 0)), None

        if c0_vel_sel is not None:
            c0_vel_mean, c0_vel_sd = _compute_mean_and_sd(c0_vel_sel)
        else:
            c0_vel_mean, c0_vel_sd = np.zeros((0, 0)), None

        if c1_vel_sel is not None:
            c1_vel_mean, c1_vel_sd = _compute_mean_and_sd(c1_vel_sel)
        else:
            c1_vel_mean, c1_vel_sd = np.zeros((0, 0)), None

        # If for some reason we ended up with no subset, fall back to ALL labels/medians
        if not idx_subset or (c0_pos_mean.size == 0 and c1_pos_mean.size == 0):
            labels_sel  = beh_labels_display
            c0_pos_mean = beh_cam0_pos_med
            c1_pos_mean = beh_cam1_pos_med
            c0_vel_mean = beh_cam0_vel_med
            c1_vel_mean = beh_cam1_vel_med
            c0_pos_sd   = None
            c1_pos_sd   = None

        if NORMALIZED_DISTANCE:
            c0_pos_mean, c0_pos_sd = rcp.mean_ci95_from_trial_traces(cam0_dist_trials)
            c1_pos_mean, c1_pos_sd = rcp.mean_ci95_from_trial_traces(cam1_dist_trials)

            labels_sel = [""]

        # Apply position/velocity display toggles
        if not PLOT_POSITION:
            c0_pos_mean = None
            c1_pos_mean = None
            c0_pos_sd = None
            c1_pos_sd = None

        if not PLOT_VELOCITY:
            c0_vel_mean = None
            c1_vel_mean = None
            c0_vel_sd = None
            c1_vel_sd = None

        band_label = "95% CI" if NORMALIZED_DISTANCE else "std"
        title_MWT_kin = f"Kinematics (mean ± {band_label}) / n={n_events} events"
        title_MWT_neural = f"Neural Activity (mean Δ) / Referenced to first {int(NORMALIZE_FIRST_MS)} ms)"

        rcp.stacked_heatmaps_plus_behv(
            NPRW_med, UA_med,          # same med heatmaps; metric is just for behavior here
            NPRW_rel_t if (NPRW_med.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_med.size   and UA_rel_t.size)   else None,
            NPRW_width_ms,
            UA_width_ms,
            out_path_2,
            title_MWT_kin,
            title_MWT_neural,
            beh_pos_ylabel=beh_pos_ylabel_for_plot,
            cmap=COLORMAP,
            cb_label_nprw=cb_label,
            cb_label_ua=cb_label,
            vmin_nprw=VMIN_NPRW, vmax_nprw=VMAX_NPRW,
            vmin_ua={
                "M1i": VMIN_UA,
                "M1s": VMIN_UA,
                "PMd": VMIN_UA,
                "SMA": VMIN_UA,
            },
            vmax_ua={
                "M1i": VMAX_UA,
                "M1s": VMAX_UA,
                "PMd": VMAX_UA,
                "SMA": VMAX_UA,
            },
            probe=nprw_probe,
            probe_locs=locs,
            stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="region_then_elec",
            beh_rel_time=beh_time_for_both,
            beh_cam0_pos=c0_pos_mean,
            beh_cam1_pos=c1_pos_mean,
            beh_cam0_vel=c0_vel_mean,
            beh_cam1_vel=c1_vel_mean,
            beh_cam0_pos_stds=c0_pos_sd,
            beh_cam1_pos_stds=c1_pos_sd,
            beh_cam0_vel_stds=c0_vel_sd,
            beh_cam1_vel_stds=c1_vel_sd,
            target_pos_cam0=target_pos_cam0_for_plot,
            target_pos_cam1=target_pos_cam1_for_plot,
            beh_labels=labels_sel,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=full_overall_title,
            beh_ylim=beh_ylim_for_plot,
            beh_ratio=BEH_RATIO,
            ch_ratio_per_row=CH_RATIO_PER_ROW,
            min_heatmap_ratio=MIN_HEATMAP_RATIO,
            ua_compact_factor=UA_COMPACT_FACTOR,
            nprw_scale=NPRW_SCALE,
            gap_beh_nprw=GAP_BEH_NPRW,
            fig_width_in=FIG_WIDTH_IN,
            height_per_ratio_in=HEIGHT_PER_RATIO_IN,
            probe_gap_ratio=PROBE_GAP_RATIO,
            probe_width_ratio=PROBE_WIDTH_RATIO,
            nprw_blank_ms=nprw_blank_ms,
            stim_dur_ms=None if is_control else stim_dur_ms,
        )
        # -----------------------------------------------------------------
        # FIGURE 4: MWT ONLY VARIANCE (same behavior subset)
        # -----------------------------------------------------------------
        out_dir_2b_parent = FIG.peri_var_meanMWT / target_label
        out_dir_2b_parent.mkdir(parents=True, exist_ok=True)
        out_path_2b = out_dir_2b_parent / f"{file_name}__var_MWT.svg"

        title_MWT_neural_var = f"Neural Variance (across {n_events} events)"

        rcp.stacked_heatmaps_plus_behv(
            NPRW_var, UA_var,
            NPRW_rel_t if (NPRW_var.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_var.size   and UA_rel_t.size)   else None,
            NPRW_width_ms,
            UA_width_ms,
            out_path_2b,
            title_MWT_kin,          # same behavior + labels as mean/STD figure
            title_MWT_neural_var,
            beh_pos_ylabel=beh_pos_ylabel_for_plot,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW_VAR, vmax_nprw=VMAX_NPRW_VAR,
            vmin_ua={
                "M1i": VMIN_UA_VAR,
                "M1s": VMIN_UA_VAR,
                "PMd": VMIN_UA_VAR,
                "SMA": VMIN_UA_VAR,
            },
            vmax_ua={
                "M1i": VMAX_UA_VAR,
                "M1s": VMAX_UA_VAR,
                "PMd": VMAX_UA_VAR,
                "SMA": VMAX_SMA_VAR,
            },
            probe=nprw_probe,
            probe_locs=locs,
            stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="region_then_elec",
            beh_rel_time=beh_time_for_both,
            beh_cam0_pos=c0_pos_mean,
            beh_cam1_pos=c1_pos_mean,
            beh_cam0_vel=c0_vel_mean,
            beh_cam1_vel=c1_vel_mean,
            beh_cam0_pos_stds=c0_pos_sd,
            beh_cam1_pos_stds=c1_pos_sd,
            beh_cam0_vel_stds=c0_vel_sd,
            beh_cam1_vel_stds=c1_vel_sd,
            target_pos_cam0=target_pos_cam0_for_plot,
            target_pos_cam1=target_pos_cam1_for_plot,
            beh_labels=labels_sel,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=full_overall_title,
            beh_ylim=beh_ylim_for_plot,
            beh_ratio=BEH_RATIO,
            ch_ratio_per_row=CH_RATIO_PER_ROW,
            min_heatmap_ratio=MIN_HEATMAP_RATIO,
            ua_compact_factor=UA_COMPACT_FACTOR,
            nprw_scale=NPRW_SCALE,
            gap_beh_nprw=GAP_BEH_NPRW,
            fig_width_in=FIG_WIDTH_IN,
            height_per_ratio_in=HEIGHT_PER_RATIO_IN,
            probe_gap_ratio=PROBE_GAP_RATIO,
            probe_width_ratio=PROBE_WIDTH_RATIO,
            nprw_blank_ms=nprw_blank_ms,
            stim_dur_ms=None if is_control else stim_dur_ms,
        )
        # -----------------------------------------------------------------
        # FIGURE 5: MWT ONLY MEDIAN BIN COUNTS (same behavior subset as MWT fig)
        # -----------------------------------------------------------------
        out_dir_counts_MWT_parent = FIG.peri_counts_meanMWT / target_label
        out_dir_counts_MWT_parent.mkdir(parents=True, exist_ok=True)
        out_path_counts_MWT = out_dir_counts_MWT_parent / f"{file_name}__counts_MWT.svg"

        title_MWT_neural_counts = (
            f"Median spike counts per bin (across {n_events} events)"
        )

        rcp.stacked_heatmaps_plus_behv(
            NPRW_med_counts, UA_med_counts,
            NPRW_rel_t if (NPRW_med_counts.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_med_counts.size   and UA_rel_t.size)   else None,
            NPRW_width_ms,
            UA_width_ms,
            out_path_counts_MWT,
            title_MWT_kin,              # MWT-only behavior panel (subset labels)
            title_MWT_neural_counts,
            beh_pos_ylabel=beh_pos_ylabel_for_plot,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW_COUNTS, vmax_nprw=VMAX_NPRW_COUNTS,
            vmin_ua={
                "M1i": VMIN_UA_COUNTS,
                "M1s": VMIN_UA_COUNTS,
                "PMd": VMIN_UA_COUNTS,
                "SMA": VMIN_UA_COUNTS,
            },
            vmax_ua={
                "M1i": VMAX_UA_COUNTS,
                "M1s": VMAX_UA_COUNTS,
                "PMd": VMAX_UA_COUNTS,
                "SMA": VMAX_UA_COUNTS,
            },
            probe=nprw_probe,
            probe_locs=locs,
            stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="region_then_elec",
            beh_rel_time=beh_time_for_both,
            beh_cam0_pos=c0_pos_mean,
            beh_cam1_pos=c1_pos_mean,
            beh_cam0_vel=c0_vel_mean,
            beh_cam1_vel=c1_vel_mean,
            beh_cam0_pos_stds=c0_pos_sd,
            beh_cam1_pos_stds=c1_pos_sd,
            beh_cam0_vel_stds=c0_vel_sd,
            beh_cam1_vel_stds=c1_vel_sd,
            target_pos_cam0=target_pos_cam0_for_plot,
            target_pos_cam1=target_pos_cam1_for_plot,
            beh_labels=labels_sel,      # MWT-only labels
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=full_overall_title,
            beh_ylim=beh_ylim_for_plot,
            beh_ratio=BEH_RATIO,
            ch_ratio_per_row=CH_RATIO_PER_ROW,
            min_heatmap_ratio=MIN_HEATMAP_RATIO,
            ua_compact_factor=UA_COMPACT_FACTOR,
            nprw_scale=NPRW_SCALE,
            gap_beh_nprw=GAP_BEH_NPRW,
            fig_width_in=FIG_WIDTH_IN,
            height_per_ratio_in=HEIGHT_PER_RATIO_IN,
            probe_gap_ratio=PROBE_GAP_RATIO,
            probe_width_ratio=PROBE_WIDTH_RATIO,
            nprw_blank_ms=nprw_blank_ms,
            stim_dur_ms=None if is_control else stim_dur_ms,
        )

        # -----------------------------------------------------------------
        # FIGURE 7: SINGLE-TRIAL HEATMAPS (first up to 4 trials, MWT only)
        # -----------------------------------------------------------------
        if NPRW_rates_zeroed.ndim == 3:
            n_trials = NPRW_rates_zeroed.shape[0]
            n_single = min(4, n_trials)

            # MWT subset (Middle + Wrist); fall back to ALL if nothing found
            idx_subset = rcp.get_subset_indices(beh_labels_display, keys=KIN_KEYPOINT_INCLUDE)
            if idx_subset:
                labels_MWT = [beh_labels_display[i] for i in idx_subset
                              if 0 <= i < len(beh_labels_display)]
            else:
                labels_MWT = beh_labels_display
                idx_subset = list(range(len(beh_labels_display)))

            def _select_single_trial(arr, idxs):
                """
                arr : (K, T) or None
                Returns arr_subset : (K', T) with K' = len(idxs) (if possible).
                If arr is None/empty or idxs invalid, returns arr unchanged.
                """
                if arr is None or arr.size == 0 or not idxs:
                    return arr
                arr = np.asarray(arr, float)
                D = arr.shape[0]
                valid = [i for i in idxs if 0 <= i < D]
                if not valid:
                    return arr
                return arr[valid, :]

            single_dir = FIG.peri_single_trials / target_label
            single_dir.mkdir(parents=True, exist_ok=True)

            for i_trial in range(n_single):
                nprw_single = NPRW_rates_zeroed[i_trial, :, :]  # (n_ch, T)
                ua_single = (
                    UA_rates_zeroed[i_trial, :, :]
                    if UA_rates_zeroed.ndim == 3 and UA_rates_zeroed.shape[0] > i_trial
                    else np.zeros((0, 0), float)
                )

                # Per-trial behavior if possible; otherwise fall back to medians
                if cam0_pos_segs is not None and cam0_pos_segs.size and i_trial < cam0_pos_segs.shape[0]:
                    c0_pos_single = np.asarray(cam0_pos_segs[i_trial], float)  # (K, T)
                    c0_vel_single = np.asarray(cam0_vel_segs[i_trial], float)
                else:
                    c0_pos_single = beh_cam0_pos_med
                    c0_vel_single = beh_cam0_vel_med

                if cam1_pos_segs is not None and cam1_pos_segs.size and i_trial < cam1_pos_segs.shape[0]:
                    c1_pos_single = np.asarray(cam1_pos_segs[i_trial], float)
                    c1_vel_single = np.asarray(cam1_vel_segs[i_trial], float)
                else:
                    c1_pos_single = beh_cam1_pos_med
                    c1_vel_single = beh_cam1_vel_med

                # --- MWT-only selection on single-trial kinematics ---
                c0_pos_single = _select_single_trial(c0_pos_single, idx_subset)
                c0_vel_single = _select_single_trial(c0_vel_single, idx_subset)
                c1_pos_single = _select_single_trial(c1_pos_single, idx_subset)
                c1_vel_single = _select_single_trial(c1_vel_single, idx_subset)

                if NORMALIZED_DISTANCE:
                    c0_pos_single = (
                        cam0_dist_trials[i_trial, :][None, :]
                        if cam0_dist_trials is not None and i_trial < cam0_dist_trials.shape[0]
                        else None
                    )
                    c1_pos_single = (
                        cam1_dist_trials[i_trial, :][None, :]
                        if cam1_dist_trials is not None and i_trial < cam1_dist_trials.shape[0]
                        else None
                    )

                    labels_single = [""]
                else:
                    labels_single = labels_MWT

                if not PLOT_POSITION:
                    c0_pos_single = None
                    c1_pos_single = None

                if not PLOT_VELOCITY:
                    c0_vel_single = None
                    c1_vel_single = None

                title_single_kin = f"Kinematics (MWT, single trial {i_trial+1})"
                title_single_neural = (
                    f"Neural Activity (single trial {i_trial+1}) / "
                    f"Referenced to first {int(NORMALIZE_FIRST_MS)} ms)"
                )

                if peri_stim_npz_loc.name.startswith('baseline'):
                    base_fn = f"{peri_stim_npz_loc.name}"
                else:
                    base_fn = f"Cond_{br_idx:03d}"

                out_path_single = single_dir / f"{base_fn}__trial_{i_trial:02d}_MWT.svg"

                rcp.stacked_heatmaps_plus_behv(
                    nprw_single,
                    ua_single,
                    NPRW_rel_t if (nprw_single.size and NPRW_rel_t.size) else None,
                    UA_rel_t   if (ua_single.size   and UA_rel_t.size)   else None,
                    NPRW_width_ms,
                    UA_width_ms,
                    out_path_single,
                    title_single_kin,
                    title_single_neural,
                    beh_pos_ylabel=beh_pos_ylabel_for_plot,
                    cmap=COLORMAP,
                    vmin_nprw=VMIN_NPRW, vmax_nprw=VMAX_NPRW,
                    vmin_ua={
                        "M1i": VMIN_UA,
                        "M1s": VMIN_UA,
                        "PMd": VMIN_UA,
                        "SMA": VMIN_UA,
                    },
                    vmax_ua={
                        "M1i": VMAX_UA,
                        "M1s": VMAX_UA,
                        "PMd": VMAX_UA,
                        "SMA": VMAX_UA,
                    },
                    probe=nprw_probe,
                    probe_locs=locs,
                    stim_idx=stim_locs,
                    probe_title="NPRW probe (stim sites highlighted)",
                    ua_ids_1based=ua_ids_1based,
                    ua_sort="region_then_elec",
                    beh_rel_time=beh_time_for_both,
                    beh_cam0_pos=c0_pos_single,
                    beh_cam1_pos=c1_pos_single,
                    beh_cam0_vel=c0_vel_single,
                    beh_cam1_vel=c1_vel_single,
                    beh_cam0_pos_stds=None,
                    beh_cam1_pos_stds=None,
                    beh_cam0_vel_stds=None,
                    beh_cam1_vel_stds=None,
                    target_pos_cam0=target_pos_cam0_for_plot,
                    target_pos_cam1=target_pos_cam1_for_plot,
                    beh_labels=labels_single,
                    title_cam1="",
                    title_cam0_vel="",
                    title_cam1_vel="",
                    sess=sess,
                    overall_title=full_overall_title,
                    beh_ylim=beh_ylim_for_plot,
                    beh_ratio=BEH_RATIO,
                    ch_ratio_per_row=CH_RATIO_PER_ROW,
                    min_heatmap_ratio=MIN_HEATMAP_RATIO,
                    ua_compact_factor=UA_COMPACT_FACTOR,
                    nprw_scale=NPRW_SCALE,
                    gap_beh_nprw=GAP_BEH_NPRW,
                    fig_width_in=FIG_WIDTH_IN,
                    height_per_ratio_in=HEIGHT_PER_RATIO_IN,
                    probe_gap_ratio=PROBE_GAP_RATIO,
                    probe_width_ratio=PROBE_WIDTH_RATIO,
                    nprw_blank_ms=nprw_blank_ms,
                    stim_dur_ms=None if is_control else stim_dur_ms,
                )


if __name__ == "__main__":
    main()
