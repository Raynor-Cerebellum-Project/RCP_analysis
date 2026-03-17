from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import imageio
import io
import matplotlib
import matplotlib.pyplot as plt
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

# --- UTAH ELECTRODE MAPPING ---
def load_electrode_mapping(csv_path: Path):
    nsp_to_elec = {}
    region_grids = {}
    elec_to_region = {}

    if not csv_path.exists():
        print(f"  [Warn] Electrode mapping CSV not found: {csv_path}")
        return nsp_to_elec, region_grids, elec_to_region

    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            elec_id = int(row['ElectrodeID'])
            nsp_id = int(row['NSP_ID'])
            region = str(row['Array']).strip()
            r = int(row['GridRow'])
            c = int(row['GridCol'])

            nsp_to_elec[nsp_id] = elec_id
            elec_to_region[elec_id] = region

            if region not in region_grids:
                region_grids[region] = np.zeros((8, 8), dtype=int)
            region_grids[region][r, c] = elec_id
            
    except Exception as e:
        print(f"  [Error] Failed to load electrode mapping from {csv_path}: {e}")

    return nsp_to_elec, region_grids, elec_to_region

MAPPING_CSV = REPO_ROOT / "scripts" / "electrode_port_mapping.csv"
nsp_to_elec_global, UTAH_ELEC_GRIDS, elec_to_region_global = load_electrode_mapping(MAPPING_CSV)

def _region_from_group_name(grp_name: str) -> str:
    return grp_name.split(" (")[0].strip()

def _region_grid(region_name: str):
    aliases = {"M1 Inf": "M1i", "M1 Sup": "M1s"}
    target = aliases.get(region_name, region_name)
    return UTAH_ELEC_GRIDS.get(target, None)

def build_elec_to_data_idx(ua_ids_1based, nsp_to_elec):
    if ua_ids_1based is None or nsp_to_elec is None:
        return {}
    
    elec_to_idx = {}
    for ch_idx, nsp_id in enumerate(ua_ids_1based):
        nsp_id = int(nsp_id)
        elec_id = nsp_to_elec.get(nsp_id, -1)
        if elec_id > 0:
            elec_to_idx[elec_id] = ch_idx
    
    return elec_to_idx

def render_spatial_grid(ax, data_slice, grid_elec, elec_to_idx, vmin, vmax, smoothing=False):
    grid = np.full((8, 8), np.nan)
    points = []
    values = []
    
    for r in range(8):
        for c in range(8):
            elec_id = int(grid_elec[r, c])
            idx = elec_to_idx.get(elec_id, None)
            if idx is not None and 0 <= idx < data_slice.shape[0]:
                val = data_slice[idx]
                grid[r, c] = val
                if not np.isnan(val):
                    points.append((r, c))
                    values.append(val)

    if smoothing and len(points) >= 4:
        filled_grid = np.copy(grid)
        mask = np.isnan(filled_grid)
        
        if len(values) > 0:
            filled_grid[mask] = np.nanmean(np.asarray(values))
        
        for _ in range(10):
            prev_grid = np.copy(filled_grid)
            for r in range(8):
                for c in range(8):
                    if mask[r, c]:
                        neighbors = []
                        if r > 0: neighbors.append(prev_grid[r-1, c])
                        if r < 7: neighbors.append(prev_grid[r+1, c])
                        if c > 0: neighbors.append(prev_grid[r, c-1])
                        if c < 7: neighbors.append(prev_grid[r, c+1])
                        if neighbors:
                            filled_grid[r, c] = np.nanmean(np.asarray(neighbors))
        
        im = ax.imshow(filled_grid, cmap=COLORMAP, vmin=vmin, vmax=vmax, 
                       origin='upper', aspect='equal', interpolation='bicubic')
    else:
        im = ax.imshow(grid, cmap=COLORMAP, vmin=vmin, vmax=vmax, 
                       origin='upper', aspect='equal')
    
    return im

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
    files = sorted(PERI_ROOT.rglob("peristim__*.npz"))
    if not files:
        print(f"[warn] No peristim__*.npz found in {PERI_ROOT} or its subdirectories")
        return

    for peri_path in files:
        # mirror the folder structure in the output
        rel_parent = peri_path.parent.relative_to(PERI_ROOT)
        current_fig_root = FIG_ROOT / rel_parent
        
        current_fig = SimpleNamespace(
            peri_median  = current_fig_root / "median_fr_plots",
            peri_var     = current_fig_root / "variance_fr_plots",
            peri_counts  = current_fig_root / "median_count_plots",
            peri_single  = current_fig_root / "single_trial_fr_plots",
        )
        current_fig.peri_median.mkdir(parents=True, exist_ok=True)
        current_fig.peri_var.mkdir(parents=True, exist_ok=True)
        current_fig.peri_counts.mkdir(parents=True, exist_ok=True)
        current_fig.peri_single.mkdir(parents=True, exist_ok=True)

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
        out_path_1 = current_fig.peri_median / f"{base_fn}__median.png"
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
        out_path_2 = current_fig.peri_var / f"{base_fn}__var.png"
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
        out_path_3 = current_fig.peri_counts / f"{base_fn}__counts.png"
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

                out_single = current_fig.peri_single / f"{base_fn}__trial_{i_trial:02d}.png"
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

        # -----------------------------------------------------------------
        # FIG 5: SPATIAL GIFS (FIRING RATES)
        # -----------------------------------------------------------------
        if UA_med.size and UA_rel_t.size and ua_ids_1based is not None:
            gif_dir = current_fig_root / "GIFs"
            gif_dir.mkdir(parents=True, exist_ok=True)
            
            elec_to_idx = build_elec_to_data_idx(ua_ids_1based, nsp_to_elec_global)
            
            # Find the groups
            u_regions = set()
            for ch_idx, nsp_id in enumerate(ua_ids_1based):
                elec_id = nsp_to_elec_global.get(int(nsp_id), -1)
                reg = elec_to_region_global.get(elec_id, None)
                if reg:
                    u_regions.add(reg)
            
            # Define GIF time points (e.g. every 10ms for smooth playing)
            t_min = UA_rel_t[0]
            t_max = UA_rel_t[-1]
            t_step = 10.0
            gif_time_pts = np.arange(t_min, t_max + t_step, t_step)
            
            vmin = VMIN_UA
            vmax = VMAX_UA
            
            for region_name in ["SMA", "PMd", "M1i", "M1s"]:
                if region_name not in u_regions:
                    continue
                grid_elec = _region_grid(region_name)
                if grid_elec is None:
                    continue

                safe_region = region_name.replace(' ', '_')
                
                render_modes = [('blocky', False), ('smoothed', True)]
                
                for mode_label, use_smoothing in render_modes:
                    frames = []
                    print(f"    Generating FR GIF ({mode_label}) for {region_name} ({len(gif_time_pts)} frames)...")
                    
                    fig_gif, ax_gif = plt.subplots(figsize=(6, 5))
                    
                    for t_target in gif_time_pts:
                        t_bin_idx = int(np.argmin(np.abs(UA_rel_t - t_target)))
                        t_actual = float(UA_rel_t[t_bin_idx])
                        
                        ax_gif.clear()
                        im = render_spatial_grid(ax_gif, UA_med[:, t_bin_idx], grid_elec, elec_to_idx, vmin, vmax, smoothing=use_smoothing)
                        
                        ax_gif.set_title(f"{region_name} | Firing Rate ({mode_label.capitalize()})\nT = {t_actual:.0f} ms", fontsize=12)
                        ax_gif.set_xticks(np.arange(8))
                        ax_gif.set_yticks(np.arange(8))
                        ax_gif.set_xticklabels(np.arange(1, 9), fontsize=8)
                        ax_gif.set_yticklabels(np.arange(1, 9), fontsize=8)
                        ax_gif.set_ylabel("Row", fontsize=10)
                        ax_gif.set_xlabel("Col", fontsize=10)
                        
                        if not fig_gif.axes[1:]:
                            fig_gif.colorbar(im, ax=ax_gif, label="Δ Firing Rate")
                            
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=80)
                        buf.seek(0)
                        frames.append(imageio.v3.imread(buf))

                    plt.close(fig_gif)

                    if frames:
                        out_name = f"spatial_heatmap_FR_{base_fn}_{safe_region}_{mode_label}.gif"
                        out_path = gif_dir / out_name
                        imageio.mimsave(out_path, frames, fps=10)
                        print(f"    Saved FR GIF -> {out_path.name}")

        print(f"[plot] done: {peri_path.name}")

if __name__ == "__main__":
    main()
