from types import SimpleNamespace
from pathlib import Path

import numpy as np
import matplotlib
from scipy.io import loadmat
from probeinterface import Probe

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

matplotlib.rcParams["svg.fonttype"] = "none"

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
WIN_MS             = (-600.0, 600.0)
NORMALIZE_FIRST_MS = 150.0

VMIN_NPRW, VMAX_NPRW = -25, 200
VMIN_UA,   VMAX_UA   = -50, 150

VMIN_NPRW_VAR, VMAX_NPRW_VAR = 0.0, 40000.0
VMIN_UA_VAR,   VMAX_UA_VAR   = 0.0, 10000.0
VMAX_SMA_VAR                 = 5000.0

VMIN_NPRW_COUNTS, VMAX_NPRW_COUNTS = 0.0, 10.0
VMIN_UA_COUNTS,   VMAX_UA_COUNTS   = 0.0, 10.0

COLORMAP = "jet"

# Layout knobs (passed into stacked_heatmaps_plus_behv)
BEH_RATIO = 0.0               # <- no behavior rows in the new NPZ
CH_RATIO_PER_ROW = 0.015
MIN_HEATMAP_RATIO = 0.6
UA_COMPACT_FACTOR = 0.95
NPRW_SCALE        = 0.6
GAP_BEH_NPRW      = 0.05       # tiny spacer (behavior is absent anyway)
FIG_WIDTH_IN         = 8.0
HEIGHT_PER_RATIO_IN  = 4.0
PROBE_GAP_RATIO      = 0.15
PROBE_WIDTH_RATIO    = 0.35

# ---------------------------------------------------------------------
# ROOTS / PARAMS
# ---------------------------------------------------------------------
FIG_ROOT   = OUT_BASE / "figures" / "peristim_new_npz"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

FIG = SimpleNamespace(
    peri_median  = FIG_ROOT / "median_fr_plots",
    peri_var     = FIG_ROOT / "variance_fr_plots",
    peri_counts  = FIG_ROOT / "median_count_plots",
    peri_single  = FIG_ROOT / "single_trial_fr_plots",
)
FIG.peri_median.mkdir(parents=True, exist_ok=True)
FIG.peri_var.mkdir(parents=True, exist_ok=True)
FIG.peri_counts.mkdir(parents=True, exist_ok=True)
FIG.peri_single.mkdir(parents=True, exist_ok=True)

PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"
NPRW_AUX_DATA = OUT_BASE / "aux_data" / "NPRW"

GEOM_PATH = (
    Path(PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
    else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None)
    else rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
)

def _safe_get_scalar_str(x) -> str:
    if isinstance(x, np.ndarray) and x.shape == ():
        return str(x.item())
    return str(x)

def main():
    # ---- load probe geometry once ----
    mat_probe = loadmat(Path(GEOM_PATH))
    nprw_geom = {
        "x": mat_probe["xcoords"].ravel(),
        "y": mat_probe["ycoords"].ravel(),
    }
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

    # ---- find peri-stim NPZs (NEW FORMAT) ----
    files = sorted(PERI_ROOT.glob("peristim__*.npz"))
    if not files:
        print(f"[warn] No peristim__*.npz found in {PERI_ROOT}")
        return

    for peri_path in files:
        peri = np.load(peri_path, allow_pickle=True)

        sess = _safe_get_scalar_str(peri["sess"])
        br_idx = int(peri["br_idx"]) if "br_idx" in peri.files else -1

        overall_title = _safe_get_scalar_str(peri["overall_title"]) if "overall_title" in peri.files else ""
        n_nprw = int(peri["n_nprw"]) if "n_nprw" in peri.files else None

        # --- neural ---
        NPRW_med        = peri["NPRW_med"]        if "NPRW_med" in peri.files else np.zeros((0, 0), float)
        NPRW_var        = peri["NPRW_var"]        if "NPRW_var" in peri.files else np.zeros((0, 0), float)
        NPRW_med_counts = peri["NPRW_med_counts"] if "NPRW_med_counts" in peri.files else np.zeros((0, 0), float)
        NPRW_rel_t      = peri["NPRW_rel_t"]      if "NPRW_rel_t" in peri.files else np.array([], float)
        NPRW_edges_ms   = peri["NPRW_edges_ms"]   if "NPRW_edges_ms" in peri.files else np.array([], float)

        UA_med        = peri["UA_med"]        if "UA_med" in peri.files else np.zeros((0, 0), float)
        UA_var        = peri["UA_var"]        if "UA_var" in peri.files else np.zeros((0, 0), float)
        UA_med_counts = peri["UA_med_counts"] if "UA_med_counts" in peri.files else np.zeros((0, 0), float)
        UA_rel_t      = peri["UA_rel_t"]      if "UA_rel_t" in peri.files else np.array([], float)
        UA_edges_ms   = peri["UA_edges_ms"]   if "UA_edges_ms" in peri.files else np.array([], float)

        ua_ids_1based = peri["ua_ids_1based"] if "ua_ids_1based" in peri.files else None

        # single-trial arrays (optional)
        NPRW_rates_zeroed = peri["NPRW_rates_zeroed"] if "NPRW_rates_zeroed" in peri.files else np.zeros((0, 0, 0), float)
        UA_rates_zeroed   = peri["UA_rates_zeroed"]   if "UA_rates_zeroed" in peri.files else np.zeros((0, 0, 0), float)

        # Titles / filenames
        if n_nprw is None:
            # fall back: infer from stim_ms_nprw if present
            n_nprw = int(peri["stim_ms_nprw"].size) if "stim_ms_nprw" in peri.files else (NPRW_rates_zeroed.shape[0] if NPRW_rates_zeroed.ndim == 3 else 0)

        base_kin_title   = f"(no behavior in new NPZ) / n={n_nprw} events"
        base_neural_title = f"Neural Activity (median Δ) / Referenced to first {int(NORMALIZE_FIRST_MS)} ms"
        full_overall_title = overall_title

        base_fn = f"Cond_{br_idx:03d}" if br_idx >= 0 else peri_path.stem

        # ---- stim site detection (probe inset) ---- TODO this can be extracted from active_channels
        stim_npz = NPRW_AUX_DATA / f"{sess}_Intan_streams" / "stim_stream.npz"
        stim_locs = None
        if stim_npz.exists():
            try:
                stim_locs = rcp.detect_stim_channels_from_npz(stim_npz, eps=1e-12, min_edges=1)
            except Exception as e:
                print(f"[warn] stim-site detection failed for {sess}, Condition {br_idx}: {e}")

        # -----------------------------------------------------------------
        # FIG 1: MEDIAN HEATMAPS
        # -----------------------------------------------------------------
        out_path_1 = FIG.peri_median / f"{base_fn}__median.png"
        rcp.stacked_heatmaps_plus_behv(
            NPRW_med, UA_med,
            NPRW_rel_t if (NPRW_med.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_med.size   and UA_rel_t.size)   else None,
            NPRW_edges_ms, UA_edges_ms,
            out_path_1,
            base_kin_title,
            base_neural_title,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW, vmax_nprw=VMAX_NPRW,
            vmin_ua={"M1i+M1s": VMIN_UA, "PMd": VMIN_UA, "SMA": VMIN_UA},
            vmax_ua={"M1i+M1s": VMAX_UA, "PMd": VMAX_UA, "SMA": VMAX_UA},
            probe=nprw_probe,
            probe_locs=locs,
            stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="region_then_elec",

            # ---- behavior is absent in new NPZ ----
            beh_rel_time=None,
            beh_cam0_pos=None, beh_cam1_pos=None,
            beh_cam0_vel=None, beh_cam1_vel=None,
            beh_cam0_pos_stds=None, beh_cam1_pos_stds=None,
            beh_cam0_vel_stds=None, beh_cam1_vel_stds=None,
            beh_labels=[],

            sess=sess,
            overall_title=full_overall_title,
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
        # FIG 2: VARIANCE HEATMAPS
        # -----------------------------------------------------------------
        out_path_2 = FIG.peri_var / f"{base_fn}__var.png"
        title_var = f"Neural Variance (across {n_nprw} events)"
        rcp.stacked_heatmaps_plus_behv(
            NPRW_var, UA_var,
            NPRW_rel_t if (NPRW_var.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_var.size   and UA_rel_t.size)   else None,
            NPRW_edges_ms, UA_edges_ms,
            out_path_2,
            base_kin_title,
            title_var,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW_VAR, vmax_nprw=VMAX_NPRW_VAR,
            vmin_ua={"M1i+M1s": VMIN_UA_VAR, "PMd": VMIN_UA_VAR, "SMA": VMIN_UA_VAR},
            vmax_ua={"M1i+M1s": VMAX_UA_VAR, "PMd": VMAX_UA_VAR, "SMA": VMAX_SMA_VAR},
            probe=nprw_probe,
            probe_locs=locs,
            stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="region_then_elec",

            beh_rel_time=None,
            beh_cam0_pos=None, beh_cam1_pos=None,
            beh_cam0_vel=None, beh_cam1_vel=None,
            beh_cam0_pos_stds=None, beh_cam1_pos_stds=None,
            beh_cam0_vel_stds=None, beh_cam1_vel_stds=None,
            beh_labels=[],

            sess=sess,
            overall_title=full_overall_title,
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
        # FIG 3: MEDIAN BIN COUNTS
        # -----------------------------------------------------------------
        out_path_3 = FIG.peri_counts / f"{base_fn}__counts.png"
        title_counts = f"Median spike counts per bin (across {n_nprw} events)"
        rcp.stacked_heatmaps_plus_behv(
            NPRW_med_counts, UA_med_counts,
            NPRW_rel_t if (NPRW_med_counts.size and NPRW_rel_t.size) else None,
            UA_rel_t   if (UA_med_counts.size   and UA_rel_t.size)   else None,
            NPRW_edges_ms, UA_edges_ms,
            out_path_3,
            base_kin_title,
            title_counts,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW_COUNTS, vmax_nprw=VMAX_NPRW_COUNTS,
            vmin_ua={"M1i+M1s": VMIN_UA_COUNTS, "PMd": VMIN_UA_COUNTS, "SMA": VMIN_UA_COUNTS},
            vmax_ua={"M1i+M1s": VMAX_UA_COUNTS, "PMd": VMAX_UA_COUNTS, "SMA": VMAX_UA_COUNTS},
            probe=nprw_probe,
            probe_locs=locs,
            stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="region_then_elec",

            beh_rel_time=None,
            beh_cam0_pos=None, beh_cam1_pos=None,
            beh_cam0_vel=None, beh_cam1_vel=None,
            beh_cam0_pos_stds=None, beh_cam1_pos_stds=None,
            beh_cam0_vel_stds=None, beh_cam1_vel_stds=None,
            beh_labels=[],

            sess=sess,
            overall_title=full_overall_title,
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
        # FIG 4: SINGLE-TRIAL HEATMAPS (up to 4)
        # -----------------------------------------------------------------
        if NPRW_rates_zeroed.ndim == 3 and NPRW_rates_zeroed.shape[0] > 0:
            n_single = min(4, NPRW_rates_zeroed.shape[0])
            for i_trial in range(n_single):
                nprw_single = NPRW_rates_zeroed[i_trial, :, :]
                ua_single = (
                    UA_rates_zeroed[i_trial, :, :]
                    if UA_rates_zeroed.ndim == 3 and UA_rates_zeroed.shape[0] > i_trial
                    else np.zeros((0, 0), float)
                )

                out_single = FIG.peri_single / f"{base_fn}__trial_{i_trial:02d}.png"
                title_single_kin = f"(no behavior) single trial {i_trial+1}"
                title_single_neural = (
                    f"Neural Activity (single trial {i_trial+1}) / "
                    f"Referenced to first {int(NORMALIZE_FIRST_MS)} ms"
                )

                rcp.stacked_heatmaps_plus_behv(
                    nprw_single, ua_single,
                    NPRW_rel_t if (nprw_single.size and NPRW_rel_t.size) else None,
                    UA_rel_t   if (ua_single.size   and UA_rel_t.size)   else None,
                    NPRW_edges_ms, UA_edges_ms,
                    out_single,
                    title_single_kin,
                    title_single_neural,
                    cmap=COLORMAP,
                    vmin_nprw=VMIN_NPRW, vmax_nprw=VMAX_NPRW,
                    vmin_ua={"M1i+M1s": VMIN_UA, "PMd": VMIN_UA, "SMA": VMIN_UA},
                    vmax_ua={"M1i+M1s": VMAX_UA, "PMd": VMAX_UA, "SMA": VMAX_UA},
                    probe=nprw_probe,
                    probe_locs=locs,
                    stim_idx=stim_locs,
                    probe_title="NPRW probe (stim sites highlighted)",
                    ua_ids_1based=ua_ids_1based,
                    ua_sort="region_then_elec",

                    beh_rel_time=None,
                    beh_cam0_pos=None, beh_cam1_pos=None,
                    beh_cam0_vel=None, beh_cam1_vel=None,
                    beh_cam0_pos_stds=None, beh_cam1_pos_stds=None,
                    beh_cam0_vel_stds=None, beh_cam1_vel_stds=None,
                    beh_labels=[],

                    sess=sess,
                    overall_title=full_overall_title,
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

        print(f"[plot] done: {peri_path.name}")

if __name__ == "__main__":
    main()
