from types import SimpleNamespace

from scipy.io import loadmat
from probeinterface import Probe

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# Local plotting settings
matplotlib.rcParams["svg.fonttype"] = "none"

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
WIN_MS             = (-600.0, 600.0)
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

COLORMAP = "jet"

# Kinematics
KINEMATICS_YLIM = (-4, 4)    # fixed y-limits for all figures
GAUSS_SMOOTH_MS = 0          # 0 → no extra smoothing here

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
FIG_ROOT   = OUT_BASE / "figures/shaded_BT_png"; FIG_ROOT.mkdir(parents=True, exist_ok=True)
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

def _strip_cam_prefix(n: str) -> str:
    """Turn 'cam0_wrist_x' → 'wrist_x', etc."""
    n = str(n)
    if n.startswith("cam0_"):
        return n[len("cam0_"):]
    if n.startswith("cam1_"):
        return n[len("cam1_"):]
    return n

def _simple_beh_labels(names: list[str],
                       keypoints: tuple[str, ...] = KEYPOINTS_ORDER) -> list[str]:
    """
    Map raw DLC-style names like:
      'DLC_Resnet50_..._wrist_x'  -> 'Wrist X'
      'cam0_elbow_y'              -> 'Elbow Y'
    If a keypoint/axis can’t be found, fall back to the original name.
    """
    out: list[str] = []
    kps_lc = tuple(kp.lower() for kp in keypoints)
    for n in names:
        s = str(n).lower()
        kp = next((kp for kp in kps_lc if kp in s), None)
        axis = "x" if ("_x" in s or s.endswith("x")) else (
               "y" if ("_y" in s or s.endswith("y")) else None)
        if kp and axis:
            # Capitalize nicely: "wrist_x" -> "Wrist X"
            base = kp.replace("_", " ").title()
            ax   = axis.upper()
            out.append(f"{base} {ax}")
        else:
            out.append(str(n))
    return out

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

def _get_subset_indices(labels: list[str], mode: str = "MWT") -> list[int]:
    """
    For "MWT" mode, keep only Middle, Wrist keypoints.
    labels : list of pretty labels ("Wrist X", "Middle Finger Y", ...)
    """
    if mode.upper() == "ALL":
        return list(range(len(labels)))

    keys = ("middle", "wrist")
    idxs = [i for i, lab in enumerate(labels) if any(k in lab.lower() for k in keys)]
    return idxs

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
    # Search both Target_A and Target_B folders
    # -----------------------------------------------------------------
    target_dirs = [PERI_ROOT / "Target_A", PERI_ROOT / "Target_B"]
    files: list[Path] = []
    for d in target_dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.npz")))

    if not files:
        print(f"[warn] No .npz files found in {target_dirs}")
        return

    for k, peri_stim_npz_loc in enumerate(files):
        # if k <= 2:
        #     continue
        if not peri_stim_npz_loc.exists():
            continue

        target_label   = peri_stim_npz_loc.parent.name
        
        # --- PRE-LOAD SKIP CHECK ---
        if SKIP_EXISTING:
            out_dir_check = FIG.peri_posvel_median / target_label
            
            # Derive expected output filename from input filename
            # Pattern: peristim__{session}__BR_{br_idx}_...
            fname = peri_stim_npz_loc.name
            check_file = None
            
            if fname.startswith('baseline'):
                check_file = f"{fname}__median_ALL.png"
            else:
                # Try to extract BR index
                import re
                match = re.search(r"BR_(\d+)", fname)
                if match:
                    br_val = int(match.group(1))
                    check_file = f"Cond_{br_val:03d}__median_ALL.png"
            
            if check_file:
                check_path = out_dir_check / check_file
                if check_path.exists():
                    print(f"  [Skip] {check_file} exists (pre-load check).")
                    continue

        print(f"Processing {peri_stim_npz_loc.name} ({target_label})...")

        peri_stim_npz = np.load(peri_stim_npz_loc, allow_pickle=True)
        sess         = str(peri_stim_npz["sess"])
        br_idx       = int(peri_stim_npz["br_idx"])
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
        parent_name  = peri_stim_npz_loc.parent.name  # "Target_A" or "Target_B"
        if parent_name == "Target_A":
            target_suffix = "ta"
        elif parent_name == "Target_B":
            target_suffix = "tb"
        else:
            target_suffix = ""
            
        idx0_mask, cam0_mean_xy, cam0_mean_xy_per_trial = _compute_target_means_for_cam(
            cam0_pos_segs, cam0_names, beh_rel_t, ts_state_segs, ts_state_rel_t, target_suffix
        )
        idx1_mask, cam1_mean_xy, cam1_mean_xy_per_trial = _compute_target_means_for_cam(
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
        NPRW_edges_ms     = peri_stim_npz["NPRW_edges_ms"]
        
        n_nprw = int(peri_stim_npz["n_nprw"]) if "n_nprw" in peri_stim_npz.files else NPRW_med.shape[0]

        UA_rates_zeroed = (
            peri_stim_npz["UA_rates_zeroed"]
            if "UA_rates_zeroed" in peri_stim_npz.files
            else np.zeros((0, 0, 0), float)
        )
        UA_med        = peri_stim_npz["UA_med"]
        UA_var        = peri_stim_npz["UA_var"]
        UA_med_counts = peri_stim_npz["UA_med_counts"]
        UA_rel_t      = peri_stim_npz["UA_rel_t"]
        UA_edges_ms   = peri_stim_npz["UA_edges_ms"]
        ua_ids_1based = peri_stim_npz["ua_ids_1based"] if "ua_ids_1based" in peri_stim_npz.files else None

        ts_state_segs = peri_stim_npz["ts_state_segs"]
            
        # -----------------------------------------------------------------
        # Labels
        # -----------------------------------------------------------------
        if cam0_names:
            beh_labels_raw = [_strip_cam_prefix(n) for n in cam0_names]
        elif cam1_names:
            beh_labels_raw = [_strip_cam_prefix(n) for n in cam1_names]
        else:
            beh_labels_raw = []

        beh_labels_display = _simple_beh_labels(beh_labels_raw, KEYPOINTS_ORDER)
        beh_time_for_both = beh_rel_t

        # Titles
        base_kin_title = f"Kinematics / n={n_nprw} events"
        base_neural_title = f"Neural Activity (median Δ) / Referenced to first {int(NORMALIZE_FIRST_MS)} ms)"
        full_overall_title = f"{overall_title} {target_label.replace('_', ' ')}"

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
        out_path_1 = out_dir_1_parent / f"{file_name}__median_ALL.png"

        if SKIP_EXISTING and out_path_1.exists():
            print(f"  [Skip] {file_name} exists.")
            continue

        rcp.stacked_heatmaps_plus_behv(
            NPRW_med, UA_med,
            NPRW_rel_t if (NPRW_med.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_med.size   and UA_rel_t.size)   else None,
            NPRW_edges_ms,
            UA_edges_ms,
            out_path_1,
            base_kin_title,
            base_neural_title,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW, vmax_nprw=VMAX_NPRW,
            vmin_ua={
                "M1i+M1s": VMIN_UA,
                "PMd":     VMIN_UA,
                "SMA":     VMIN_UA,
            },
            vmax_ua={
                "M1i+M1s": VMAX_UA,
                "PMd":     VMAX_UA,
                "SMA":     VMAX_UA,
            },
            probe=nprw_probe,
            probe_locs=locs,
            stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="region_then_elec",
            beh_rel_time=beh_time_for_both,
            beh_cam0_pos=beh_cam0_pos_med,
            beh_cam1_pos=beh_cam1_pos_med,
            beh_cam0_vel=beh_cam0_vel_med,
            beh_cam1_vel=beh_cam1_vel_med,
            beh_cam0_pos_stds=None,  # no shading for median figure
            beh_cam1_pos_stds=None,
            beh_cam0_vel_stds=None,
            beh_cam1_vel_stds=None,
            target_pos_cam0=target_pos_cam0,
            target_pos_cam1=target_pos_cam1,
            beh_labels=beh_labels_display,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=full_overall_title,
            beh_ylim=KINEMATICS_YLIM,
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
        )
        # -----------------------------------------------------------------
        # FIGURE 2: ALL VARIANCE TRACES (no shading, all keypoints)
        # -----------------------------------------------------------------
        out_dir_1b_parent = FIG.peri_var_median / target_label
        out_dir_1b_parent.mkdir(parents=True, exist_ok=True)
        out_path_1b = out_dir_1b_parent / f"{file_name}__var_ALL.png"

        base_neural_var_title = f"Neural Variance (across {n_nprw} events)"

        rcp.stacked_heatmaps_plus_behv(
            NPRW_var, UA_var,
            NPRW_rel_t if (NPRW_var.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_var.size   and UA_rel_t.size)   else None,
            NPRW_edges_ms,
            UA_edges_ms,
            out_path_1b,
            base_kin_title,   # same behavior panel as median
            base_neural_var_title,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW_VAR, vmax_nprw=VMAX_NPRW_VAR,
            vmin_ua={
                "M1i+M1s": VMIN_UA_VAR,
                "PMd":     VMIN_UA_VAR,
                "SMA":     VMIN_UA_VAR,
            },
            vmax_ua={
                "M1i+M1s": VMAX_UA_VAR,
                "PMd":     VMAX_UA_VAR,
                "SMA":     VMAX_SMA_VAR,  # use tighter cap for SMA if you like
            },
            probe=nprw_probe,
            probe_locs=locs,
            stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="region_then_elec",
            beh_rel_time=beh_time_for_both,
            beh_cam0_pos=beh_cam0_pos_med,
            beh_cam1_pos=beh_cam1_pos_med,
            beh_cam0_vel=beh_cam0_vel_med,
            beh_cam1_vel=beh_cam1_vel_med,
            beh_cam0_pos_stds=None,
            beh_cam1_pos_stds=None,
            beh_cam0_vel_stds=None,
            beh_cam1_vel_stds=None,
            target_pos_cam0=target_pos_cam0,
            target_pos_cam1=target_pos_cam1,
            beh_labels=beh_labels_display,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=full_overall_title,
            beh_ylim=KINEMATICS_YLIM,
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
        )

        # -----------------------------------------------------------------
        # FIGURE 3: MWT ONLY (Mean + shading)
        # -----------------------------------------------------------------
        out_dir_2_parent = FIG.peri_posvel_meanMWT / target_label
        out_dir_2_parent.mkdir(parents=True, exist_ok=True)
        out_path_2 = out_dir_2_parent / f"{file_name}__mean_MWT.png"

        # Select MWT indices
        idx_subset = _get_subset_indices(beh_labels_display, mode="MWT")

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

        title_MWT_kin = (f"Kinematics (mean ± std) / n={n_nprw} events")
        title_MWT_neural = f"Neural Activity (mean Δ) / Referenced to first {int(NORMALIZE_FIRST_MS)} ms)"

        rcp.stacked_heatmaps_plus_behv(
            NPRW_med, UA_med,          # same med heatmaps; metric is just for behavior here
            NPRW_rel_t if (NPRW_med.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_med.size   and UA_rel_t.size)   else None,
            NPRW_edges_ms,
            UA_edges_ms,
            out_path_2,
            title_MWT_kin,
            title_MWT_neural,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW, vmax_nprw=VMAX_NPRW,
            vmin_ua={
                "M1i+M1s": VMIN_UA,
                "PMd":     VMIN_UA,
                "SMA":     VMIN_UA,
            },
            vmax_ua={
                "M1i+M1s": VMAX_UA,
                "PMd":     VMAX_UA,
                "SMA":     VMAX_UA,
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
            target_pos_cam0=target_pos_cam0,
            target_pos_cam1=target_pos_cam1,
            beh_labels=labels_sel,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=full_overall_title,
            beh_ylim=KINEMATICS_YLIM,
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
        )
        # -----------------------------------------------------------------
        # FIGURE 4: MWT ONLY VARIANCE (same behavior subset)
        # -----------------------------------------------------------------
        out_dir_2b_parent = FIG.peri_var_meanMWT / target_label
        out_dir_2b_parent.mkdir(parents=True, exist_ok=True)
        out_path_2b = out_dir_2b_parent / f"{file_name}__var_MWT.png"

        title_MWT_neural_var = f"Neural Variance (across {n_nprw} events)"

        rcp.stacked_heatmaps_plus_behv(
            NPRW_var, UA_var,
            NPRW_rel_t if (NPRW_var.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_var.size   and UA_rel_t.size)   else None,
            NPRW_edges_ms,
            UA_edges_ms,
            out_path_2b,
            title_MWT_kin,          # same behavior + labels as mean/STD figure
            title_MWT_neural_var,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW_VAR, vmax_nprw=VMAX_NPRW_VAR,
            vmin_ua={
                "M1i+M1s": VMIN_UA_VAR,
                "PMd":     VMIN_UA_VAR,
                "SMA":     VMIN_UA_VAR,
            },
            vmax_ua={
                "M1i+M1s": VMAX_UA_VAR,
                "PMd":     VMAX_UA_VAR,
                "SMA":     VMAX_SMA_VAR,
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
            target_pos_cam0=target_pos_cam0,
            target_pos_cam1=target_pos_cam1,
            beh_labels=labels_sel,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=full_overall_title,
            beh_ylim=KINEMATICS_YLIM,
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
        )
        # -----------------------------------------------------------------
        # FIGURE 5: MWT ONLY MEDIAN BIN COUNTS (same behavior subset as MWT fig)
        # -----------------------------------------------------------------
        out_dir_counts_MWT_parent = FIG.peri_counts_meanMWT / target_label
        out_dir_counts_MWT_parent.mkdir(parents=True, exist_ok=True)
        out_path_counts_MWT = out_dir_counts_MWT_parent / f"{file_name}__counts_MWT.png"

        title_MWT_neural_counts = (
            f"Median spike counts per bin (across {n_nprw} events)"
        )

        rcp.stacked_heatmaps_plus_behv(
            NPRW_med_counts, UA_med_counts,
            NPRW_rel_t if (NPRW_med_counts.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_med_counts.size   and UA_rel_t.size)   else None,
            NPRW_edges_ms,
            UA_edges_ms,
            out_path_counts_MWT,
            title_MWT_kin,              # MWT-only behavior panel (subset labels)
            title_MWT_neural_counts,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW_COUNTS, vmax_nprw=VMAX_NPRW_COUNTS,
            vmin_ua={
                "M1i+M1s": VMIN_UA_COUNTS,
                "PMd":     VMIN_UA_COUNTS,
                "SMA":     VMIN_UA_COUNTS,
            },
            vmax_ua={
                "M1i+M1s": VMAX_UA_COUNTS,
                "PMd":     VMAX_UA_COUNTS,
                "SMA":     VMAX_UA_COUNTS,
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
            target_pos_cam0=target_pos_cam0,
            target_pos_cam1=target_pos_cam1,
            beh_labels=labels_sel,      # MWT-only labels
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=full_overall_title,
            beh_ylim=KINEMATICS_YLIM,
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
        )

        # -----------------------------------------------------------------
        # FIGURE 7: SINGLE-TRIAL HEATMAPS (first up to 4 trials, MWT only)
        # -----------------------------------------------------------------
        if NPRW_rates_zeroed.ndim == 3:
            n_trials = NPRW_rates_zeroed.shape[0]
            n_single = min(4, n_trials)

            # MWT subset (Middle + Wrist); fall back to ALL if nothing found
            idx_subset = _get_subset_indices(beh_labels_display, mode="MWT")
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

                title_single_kin = f"Kinematics (MWT, single trial {i_trial+1})"
                title_single_neural = (
                    f"Neural Activity (single trial {i_trial+1}) / "
                    f"Referenced to first {int(NORMALIZE_FIRST_MS)} ms)"
                )

                if peri_stim_npz_loc.name.startswith('baseline'):
                    base_fn = f"{peri_stim_npz_loc.name}"
                else:
                    base_fn = f"Cond_{br_idx:03d}"

                out_path_single = single_dir / f"{base_fn}__trial_{i_trial:02d}_MWT.png"

                rcp.stacked_heatmaps_plus_behv(
                    nprw_single,
                    ua_single,
                    NPRW_rel_t if (nprw_single.size and NPRW_rel_t.size) else None,
                    UA_rel_t   if (ua_single.size   and UA_rel_t.size)   else None,
                    NPRW_edges_ms,
                    UA_edges_ms,
                    out_path_single,
                    title_single_kin,
                    title_single_neural,
                    cmap=COLORMAP,
                    vmin_nprw=VMIN_NPRW, vmax_nprw=VMAX_NPRW,
                    vmin_ua={
                        "M1i+M1s": VMIN_UA,
                        "PMd":     VMIN_UA,
                        "SMA":     VMIN_UA,
                    },
                    vmax_ua={
                        "M1i+M1s": VMAX_UA,
                        "PMd":     VMAX_UA,
                        "SMA":     VMAX_UA,
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
                    target_pos_cam0=target_pos_cam0,
                    target_pos_cam1=target_pos_cam1,
                    beh_labels=labels_MWT,
                    title_cam1="",
                    title_cam0_vel="",
                    title_cam1_vel="",
                    sess=sess,
                    overall_title=full_overall_title,
                    beh_ylim=KINEMATICS_YLIM,
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
                )


if __name__ == "__main__":
    main()
