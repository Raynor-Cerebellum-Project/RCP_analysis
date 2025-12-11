from pathlib import Path
from typing import Tuple
from types import SimpleNamespace
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
matplotlib.rcParams["svg.fonttype"] = "none"

from probeinterface import Probe
from scipy.io import loadmat
import RCP_analysis as rcp

# ---------- CONFIG ----------
WIN_MS             = (-600.0, 600.0)
NORMALIZE_FIRST_MS = 150.0

# Baseline-friendly color ranges
VMIN_NPRW_BASELINE, VMAX_NPRW_BASELINE = -25, 100
VMIN_UA_BASELINE,   VMAX_UA_BASELINE   = -25, 100
VMIN_SMA_BASELINE,  VMAX_SMA_BASELINE  = -10, 35

# Variance ranges (used for both baseline + peri-stim)
VMIN_NPRW_VAR, VMAX_NPRW_VAR = 0.0, 40000.0
VMIN_UA_VAR,   VMAX_UA_VAR   = 0.0, 10000.0
VMAX_SMA_VAR                  = 5000.0

# Peri-stim med ranges (a bit wider than baseline)
VMIN_NPRW_STIM, VMAX_NPRW_STIM = -25, 200
VMIN_UA_STIM,   VMAX_UA_STIM   = -50, 150

COLORMAP = "jet"

# ---------- params / roots ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(
    REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT
)

SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE    = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)
PERI_ROOT   = OUT_BASE / "checkpoints" / "PeriStim"; PERI_ROOT.mkdir(parents=True, exist_ok=True)
NPRW_AUX_DATA   = OUT_BASE / "aux_data" / "NPRW"
METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"

FIG_ROOT   = OUT_BASE / "figures"; FIG_ROOT.mkdir(parents=True, exist_ok=True)
FIG = SimpleNamespace(peri_posvel_median=FIG_ROOT / "median_fr_plots", peri_posvel_var=FIG_ROOT / "var_fr_plots")
FIG.peri_posvel_median.mkdir(parents=True, exist_ok=True)
FIG.peri_posvel_var.mkdir(parents=True, exist_ok=True)

# ---------- Impedance loading ----------
UA_IMP_MAX_KOHM = 1000.0        # threshold for excluding UA rows; adjust as needed
EXCLUDE_UA_HIGH_Z = True       # set False to disable masking quickly

# Where the files live; mirror the other script:
IMP_BASE = OUT_BASE.parents[0]  # same as your script (one level above OUT_BASE)
IMP_FILES = {
    "A": IMP_BASE / "Impedances" / "Utah_imp_Port_A",
    "B": IMP_BASE / "Impedances" / "Utah_imp_Port_B",
}

GEOM_PATH = (
    Path(PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
    else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None)
    else rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
)

# ---------- kinematics ----------
KEYPOINTS_ORDER: Tuple[str, ...] = tuple(PARAMS.kinematics.get("keypoints"))

_imp_pat_elecnum = re.compile(
    r"\belec\s*\d+\s*-\s*(\d{1,3})\s+([0-9]+(?:\.[0-9]+)?)\s*(k?ohms?|kΩ|ohms?|Ω)\b",
    flags=re.IGNORECASE,
)

def _port_from_baseline_path(npz_path: Path) -> str | None:
    """
    Extract port 'A' or 'B' from the baseline filename / path.
    Looks for 'port_A' / 'port B' / 'portA' etc.
    """
    s = npz_path.name
    m = re.search(r"port[_\s]*([AB])(?=_|\b)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None

def _build_impedance_dict_for_target() -> tuple[dict[int, float], dict[int, float]]:
    """
    Load UA impedances for both ports A and B from the TextEdit dumps.
    Returns (imp_A, imp_B) where each is {elec#: impedance_kΩ}.
    """
    def _try_load_imp(tag: str) -> dict[int, float]:
        out: dict[int, float] = {}
        p = IMP_FILES.get(tag)
        if not p:
            return out
        if not p.exists():
            print(f"[warn] UA impedance file not found for port {tag}: {p}")
            return out
        try:
            d = load_impedances_from_textedit_dump(p)
            out.update(d)
            print(f"[info] Parsed {len(d)} UA impedances from {p.name} (port {tag}).")
        except Exception as e:
            print(f"[warn] could not parse UA impedances from {p} (port {tag}): {e}")
        return out

    imp_a = _try_load_imp("A")
    imp_b = _try_load_imp("B")
    return imp_a, imp_b

def _unit_to_kohm(val: float, unit: str) -> float:
    u = (unit or "").lower()
    if "k" in u:
        return float(val)
    return float(val) / 1000.0  # Ω → kΩ

def _read_text_loose(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(str(p))
    b = p.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    # fall back to printable subset
    filtered = bytes(ch for ch in b if 9 <= ch <= 126 or ch in (10, 13))
    return filtered.decode("latin-1", errors="ignore")

def load_impedances_from_textedit_dump(path_like: str | Path) -> dict[int, float]:
    """
    Parse lines like 'elec1-5   201 kOhm' from a loose text dump.
    Returns { Elec#: impedance_kΩ }.
    """
    txt = _read_text_loose(Path(path_like))
    out: dict[int, float] = {}
    for m in _imp_pat_elecnum.finditer(txt):
        elec_str, val_str, unit = m.groups()
        try:
            elec = int(elec_str)
            val = float(val_str)
        except Exception:
            continue
        out[elec] = _unit_to_kohm(val, unit)
    return out

def _simple_beh_labels(names, keypoints: Tuple[str, ...] = KEYPOINTS_ORDER):
    """
    Map raw DLC-style names like
      'DLC_Resnet50_..._wrist_x'  -> 'wrist_x'
      'cam0_elbow_y'              -> 'elbow_y'
    If a keypoint/axis can’t be found, fall back to the original name.
    """
    out = []
    kps_lc = tuple(kp.lower() for kp in keypoints)
    for n in names:
        s = str(n).lower()
        kp = next((kp for kp in kps_lc if kp in s), None)
        axis = "x" if ("_x" in s or s.endswith("x")) else (
               "y" if ("_y" in s or s.endswith("y")) else None)
        if kp and axis:
            out.append(f"{kp}_{axis}")
        else:
            out.append(str(n))
    return out

def _strip_cam_prefix(n: str) -> str:
    """
    Turn "cam0_wrist_x" -> "wrist_x",
         "cam1_middle_finger_tip_y" -> "middle_finger_tip_y"
    """
    n = str(n)
    if n.startswith("cam0_"):
        return n[len("cam0_"):]
    if n.startswith("cam1_"):
        return n[len("cam1_"):]
    return n

def _get_first_array(z, names, *, default=None, as_float=True):
    """Try a list of possible field names; return first present."""
    for nm in names:
        if nm in z.files:
            arr = z[nm]
            if as_float:
                return arr.astype(float)
            return arr
    if default is None:
        return np.zeros((0, 0), float) if as_float else None
    return np.array(default, float) if as_float else default

def _get_int(z, names, default: int = 0) -> int:
    for nm in names:
        if nm in z.files:
            v = z[nm]
            return int(v.item() if getattr(v, "shape", ()) == () else v)
    return int(default)

# ---------- baseline plotting ----------
def _plot_one_baseline_npz(npz_path: Path, target_label: str, ua_imp: dict[int, float]):
    """
    Load a baseline__*.npz from PeriStim/Target_X and plot median + var figures
    into PeriStim/Target_X/figures using stacked_heatmaps_plus_behv.
    """
    with np.load(npz_path, allow_pickle=True) as z:
        sess_array = z["sess"]
        sess = str(sess_array.item() if getattr(sess_array, "shape", ()) == () else sess_array)

        overall_title_arr = z["overall_title"]
        overall_title = str(
            overall_title_arr.item()
            if getattr(overall_title_arr, "shape", ()) == ()
            else overall_title_arr
        )

        beh_rel_t          = _get_first_array(z, ["beh_rel_t"], default=np.arange(0.0))
        beh_cam0_pos_med   = _get_first_array(z, ["beh_cam0_pos_med"], default=[], as_float=True)
        beh_cam1_pos_med   = _get_first_array(z, ["beh_cam1_pos_med"], default=[], as_float=True)
        beh_cam0_vel_med   = _get_first_array(z, ["beh_cam0_vel_med"], default=[], as_float=True)
        beh_cam1_vel_med   = _get_first_array(z, ["beh_cam1_vel_med"], default=[], as_float=True)
        beh_cam0_names_raw = z["beh_cam0_names"].astype(str).tolist() if "beh_cam0_names" in z.files else []
        beh_cam1_names_raw = z["beh_cam1_names"].astype(str).tolist() if "beh_cam1_names" in z.files else []

        NPRW_med   = _get_first_array(z, ["NPRW_med", "nprw_med"])
        UA_med     = _get_first_array(z, ["UA_med", "ua_med"])
        NPRW_var   = _get_first_array(z, ["NPRW_var", "nprw_var"])
        UA_var     = _get_first_array(z, ["UA_var", "ua_var"])
        NPRW_rel_t = _get_first_array(z, ["NPRW_rel_t", "nprw_rel_t"], default=np.arange(0.0))
        UA_rel_t   = _get_first_array(z, ["UA_rel_t", "ua_rel_t"], default=np.arange(0.0))

        n_nprw     = _get_int(z, ["n_nprw"], default=NPRW_med.shape[0])
        n_nprw_v   = _get_int(z, ["n_nprw_var", "n_nprw"], default=NPRW_var.shape[0])
        ua_ids_1based = z["ua_ids_1based"] if "ua_ids_1based" in z.files else None
    
    # ---- UA impedance masking (baseline) ----
    ua_plot_med = UA_med
    ua_plot_var = UA_var

    if EXCLUDE_UA_HIGH_Z and ua_plot_med.size and ua_ids_1based is not None and ua_imp:
        try:
            ids_arr = np.asarray(ua_ids_1based)
            keep = np.ones(ids_arr.size, dtype=bool)
            any_imp = False
            for r, eid in enumerate(ids_arr):
                if eid is None or not np.isfinite(eid):
                    continue
                z_imp = ua_imp.get(int(eid))
                if z_imp is not None and np.isfinite(z_imp):
                    any_imp = True
                    if z_imp > UA_IMP_MAX_KOHM:
                        keep[r] = False

            if any_imp:
                if keep.any():
                    ua_plot_med = ua_plot_med[keep, :]
                    if ua_plot_var.size:
                        ua_plot_var = ua_plot_var[keep, :]
                    ua_ids_1based = ids_arr[keep]
                    print(
                        f"[info] Baseline UA: kept {keep.sum()}/{keep.size} rows (≤ {UA_IMP_MAX_KOHM:g} kΩ)."
                    )
                else:
                    print("[warn] Baseline UA: impedance mask would remove all rows; skipping mask.")
        except Exception as e:
            print(f"[warn] Baseline UA impedance mask failed: {e}")
            
    if beh_cam0_names_raw:
        beh_labels = _simple_beh_labels(
            [_strip_cam_prefix(n) for n in beh_cam0_names_raw],
            KEYPOINTS_ORDER,
        )
    elif beh_cam1_names_raw:
        beh_labels = _simple_beh_labels(
            [_strip_cam_prefix(n) for n in beh_cam1_names_raw],
            KEYPOINTS_ORDER,
        )
    else:
        beh_labels = []

    beh_time_for_both = beh_rel_t
    title_kinematics = f"Kinematics / Referenced to first {int(NORMALIZE_FIRST_MS)} ms"

    # ---------- MEDIAN FIG ----------
    title_NA = f"Baseline (IR-aligned): Neural Activity (median Δ across {n_nprw} events)"
    
    fig_dir = FIG.peri_posvel_median / f"Target_{target_label}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_svg_median = fig_dir / f"{npz_path.stem}.png"

    rcp.stacked_heatmaps_plus_behv(
        NPRW_med,
        ua_plot_med,
        NPRW_rel_t if NPRW_med.size else None,
        UA_rel_t   if UA_med.size   else None,
        out_svg_median,
        title_kinematics,
        title_NA,
        cmap=COLORMAP,
        vmin_nprw=VMIN_NPRW_BASELINE,
        vmax_nprw=VMAX_NPRW_BASELINE,
        vmin_ua={
            "M1i+M1s": VMIN_UA_BASELINE,
            "PMd":     VMIN_UA_BASELINE,
            "SMA":     VMIN_SMA_BASELINE,
        },
        vmax_ua={
            "M1i+M1s": VMAX_UA_BASELINE,
            "PMd":     VMAX_UA_BASELINE,
            "SMA":     VMAX_SMA_BASELINE,
        },
        probe=None,
        probe_locs=None,
        stim_idx=None,
        probe_title="",
        ua_ids_1based=ua_ids_1based,
        ua_sort="region_then_elec",
        beh_rel_time=beh_time_for_both,
        beh_cam0_lines=beh_cam0_pos_med,
        beh_cam1_lines=beh_cam1_pos_med,
        beh_cam0_vel_lines=beh_cam0_vel_med,
        beh_cam1_vel_lines=beh_cam1_vel_med,
        beh_labels=beh_labels,
        title_cam1="",
        title_cam0_vel="",
        title_cam1_vel="",
        sess=sess,
        overall_title=overall_title,
    )
    print(f"[baseline] wrote {out_svg_median}")

    # ---------- VAR FIG ----------
    if NPRW_var.size or UA_var.size:
        title_NA_var = (
            f"Baseline (IR-aligned): Neural Activity VAR "
            f"(variance of Δ across {n_nprw_v} events)"
        )
        fig_dir_var = FIG.peri_posvel_var / f"Target_{target_label}"
        fig_dir_var.mkdir(parents=True, exist_ok=True)

        out_svg_var = fig_dir_var / f"{npz_path.stem}__VAR.png"

        rcp.stacked_heatmaps_plus_behv(
            NPRW_var,
            ua_plot_var,
            NPRW_rel_t if NPRW_var.size else None,
            UA_rel_t   if UA_var.size   else None,
            out_svg_var,
            title_kinematics,
            title_NA_var,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW_VAR,
            vmax_nprw=VMAX_NPRW_VAR,
            vmin_ua={
                "M1i+M1s": VMIN_UA_VAR,
                "PMd":     VMIN_UA_VAR,
                "SMA":     0.0,
            },
            vmax_ua={
                "M1i+M1s": VMAX_UA_VAR,
                "PMd":     VMAX_UA_VAR,
                "SMA":     VMAX_SMA_VAR,
            },
            probe=None,
            probe_locs=None,
            stim_idx=None,
            probe_title="",
            ua_ids_1based=ua_ids_1based,
            ua_sort="none",
            beh_rel_time=beh_time_for_both,
            beh_cam0_lines=beh_cam0_pos_med,
            beh_cam1_lines=beh_cam1_pos_med,
            beh_cam0_vel_lines=beh_cam0_vel_med,
            beh_cam1_vel_lines=beh_cam1_vel_med,
            beh_labels=beh_labels,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=overall_title,
        )
        print(f"[baseline] wrote {out_svg_var}")

# ---------- peri-stim plotting ----------
def _plot_one_peristim_npz(npz_path: Path, target_label: str, ua_imp_a: dict[int, float], ua_imp_b: dict[int, float]):
    """
    Load a peristim__*.npz from PeriStim/Target_X and plot median + var figures
    into PeriStim/Target_X/figures.
    """
    with np.load(npz_path, allow_pickle=True) as z:
        sess_array = z["sess"]
        sess = str(sess_array.item() if getattr(sess_array, "shape", ()) == () else sess_array)

        br_idx = _get_int(z, ["br_idx"], default=-1)

        overall_title_arr = z["overall_title"]
        overall_title = str(
            overall_title_arr.item()
            if getattr(overall_title_arr, "shape", ()) == ()
            else overall_title_arr
        )

        beh_rel_t          = _get_first_array(z, ["beh_rel_t", "beh_rel_t_v"], default=np.arange(0.0))
        beh_cam0_pos_med   = _get_first_array(z, ["beh_cam0_pos_med", "cam0_pos_med"], default=[], as_float=True)
        beh_cam1_pos_med   = _get_first_array(z, ["beh_cam1_pos_med", "cam1_pos_med"], default=[], as_float=True)
        beh_cam0_vel_med   = _get_first_array(z, ["beh_cam0_vel_med", "cam0_vel_med"], default=[], as_float=True)
        beh_cam1_vel_med   = _get_first_array(z, ["beh_cam1_vel_med", "cam1_vel_med"], default=[], as_float=True)
        beh_cam0_names_raw = z["beh_cam0_names"].astype(str).tolist() if "beh_cam0_names" in z.files else []
        beh_cam1_names_raw = z["beh_cam1_names"].astype(str).tolist() if "beh_cam1_names" in z.files else []

        NPRW_med   = _get_first_array(z, ["NPRW_med", "nprw_med"])
        UA_med     = _get_first_array(z, ["UA_med", "ua_med"])
        NPRW_var   = _get_first_array(z, ["NPRW_var", "nprw_var"])
        UA_var     = _get_first_array(z, ["UA_var", "ua_var"])
        NPRW_rel_t = _get_first_array(z, ["NPRW_rel_t", "nprw_rel_t"], default=np.arange(0.0))
        UA_rel_t   = _get_first_array(z, ["UA_rel_t", "ua_rel_t"], default=np.arange(0.0))

        n_nprw     = _get_int(z, ["n_nprw"], default=NPRW_med.shape[0])
        n_nprw_v   = _get_int(z, ["n_nprw_var", "n_nprw"], default=NPRW_var.shape[0])
        ua_ids_1based = z["ua_ids_1based"] if "ua_ids_1based" in z.files else None
    
    # Figure out which port this BR file used
    ua_imp: dict[int, float] = {}
    if EXCLUDE_UA_HIGH_Z:
        try:
            br2port = rcp.get_metadata_mapping(METADATA_CSV, "BR_File", "UA_port")
            # handle int vs str keys
            port_raw = br2port.get(br_idx)
            if port_raw is None:
                port_raw = br2port.get(str(br_idx))
            if port_raw is not None:
                port = str(port_raw).strip().upper()
                if port.startswith("A"):
                    ua_imp = ua_imp_a
                elif port.startswith("B"):
                    ua_imp = ua_imp_b
                else:
                    print(f"[warn] Unknown UA_Port value '{port_raw}' for BR {br_idx}")
            else:
                print(f"[warn] No UA_Port mapping found for BR {br_idx}")
        except Exception as e:
            print(f"[warn] Could not load BR→UA_Port mapping from {METADATA_CSV}: {e}")
            ua_imp = {}
            
    # ---- UA impedance masking (peri-stim) ----
    ua_plot_med = UA_med
    ua_plot_var = UA_var

    if EXCLUDE_UA_HIGH_Z and ua_plot_med.size and ua_ids_1based is not None and ua_imp:
        try:
            ids_arr = np.asarray(ua_ids_1based)
            keep = np.ones(ids_arr.size, dtype=bool)
            any_imp = False
            for r, eid in enumerate(ids_arr):
                if eid is None or not np.isfinite(eid):
                    continue
                z_imp = ua_imp.get(int(eid))
                if z_imp is not None and np.isfinite(z_imp):
                    any_imp = True
                    if z_imp > UA_IMP_MAX_KOHM:
                        keep[r] = False

            if any_imp:
                if keep.any():
                    ua_plot_med = ua_plot_med[keep, :]
                    if ua_plot_var.size:
                        ua_plot_var = ua_plot_var[keep, :]
                    ua_ids_1based = ids_arr[keep]
                    print(
                        f"[info] Peri-stim UA (BR {br_idx}): kept {keep.sum()}/{keep.size} rows "
                        f"(≤ {UA_IMP_MAX_KOHM:g} kΩ)."
                    )
                else:
                    print("[warn] Peri-stim UA: impedance mask would remove all rows; skipping mask.")
        except Exception as e:
            print(f"[warn] Peri-stim UA impedance mask failed: {e}")

    # --- geometry / probe as you already have ---
    mat_probe = loadmat(Path(GEOM_PATH))
    nprw_geom = {}
    nprw_geom["x"] = mat_probe["xcoords"].ravel()
    nprw_geom["y"] = mat_probe["ycoords"].ravel()
    assert nprw_geom["x"].size == nprw_geom["y"].size
    if "chanMap0ind" in mat_probe:
        nprw_probe_mapping = nprw_geom["device_index_0based"] = mat_probe["chanMap0ind"].ravel()
    else:
        raise ValueError("No 0-based chanmap in .mat geometry file.")
    if nprw_probe_mapping.size != nprw_geom["x"].size:
        raise ValueError("device_index_0based length != #contacts")

    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(
        positions=np.c_[nprw_geom["x"], nprw_geom["y"]],
        shapes="square", shape_params={"width": 12.0}
    )
    nprw_probe.set_device_channel_indices(nprw_probe_mapping)
    locs = nprw_probe.contact_positions.astype(float)
    
    if beh_cam0_names_raw:
        beh_labels = _simple_beh_labels(
            [_strip_cam_prefix(n) for n in beh_cam0_names_raw],
            KEYPOINTS_ORDER,
        )
    elif beh_cam1_names_raw:
        beh_labels = _simple_beh_labels(
            [_strip_cam_prefix(n) for n in beh_cam1_names_raw],
            KEYPOINTS_ORDER,
        )
    else:
        beh_labels = []

    beh_time_for_both = beh_rel_t
    title_kinematics = f"Kinematics / Referenced to first {int(NORMALIZE_FIRST_MS)} ms"

    stim_npz = NPRW_AUX_DATA / f"{sess}_Intan_streams" / "stim_stream.npz"
    stim_locs = None
    if stim_npz.exists():
        try:
            stim_locs = rcp.detect_stim_channels_from_npz(stim_npz, eps=1e-12, min_edges=1)
        except Exception as e:
            print(f"[warn] stim-site detection failed for {sess}, Condition {br_idx}: {e}")
    # ---------- MEDIAN FIG ----------
    title_NA = (
        f"Peri-stim (stim-aligned): Neural Activity "
        f"(median Δ across {n_nprw} events, BR {br_idx:03d}, Target {target_label})"
    )
    fig_dir = FIG.peri_posvel_median / f"Target_{target_label}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out_svg_median = fig_dir / f"Cond_{br_idx:03d}__peristim.png"
    
    rcp.stacked_heatmaps_plus_behv(
        NPRW_med,
        ua_plot_med,
        NPRW_rel_t if NPRW_med.size else None,
        UA_rel_t   if UA_med.size   else None,
        
        out_svg_median,
        title_kinematics,
        title_NA,
        cmap=COLORMAP,
        vmin_nprw=VMIN_NPRW_STIM,
        vmax_nprw=VMAX_NPRW_STIM,
        vmin_ua={
            "M1i+M1s": VMIN_UA_STIM,
            "PMd":     VMIN_UA_STIM,
            "SMA":     VMIN_UA_STIM,
        },
        vmax_ua={
            "M1i+M1s": VMAX_UA_STIM,
            "PMd":     VMAX_UA_STIM,
            "SMA":     VMAX_UA_STIM,
        },
        probe=nprw_probe, probe_locs=locs, stim_idx=stim_locs,
        probe_title="NPRW probe (stim sites highlighted)",
        ua_ids_1based=ua_ids_1based,
        ua_sort="region_then_elec",
        beh_rel_time=beh_time_for_both,
        beh_cam0_lines=beh_cam0_pos_med,
        beh_cam1_lines=beh_cam1_pos_med,
        beh_cam0_vel_lines=beh_cam0_vel_med,
        beh_cam1_vel_lines=beh_cam1_vel_med,
        beh_labels=beh_labels,
        title_cam1="",
        title_cam0_vel="",
        title_cam1_vel="",
        sess=sess,
        overall_title=overall_title,
    )
    print(f"[peristim] wrote {out_svg_median}")

    # ---------- VAR FIG ----------
    if NPRW_var.size or UA_var.size:
        title_NA_var = (
            f"Peri-stim (stim-aligned): Neural Activity VAR "
            f"(variance of Δ across {n_nprw_v} events, BR {br_idx:03d}, Target {target_label})"
        )
        fig_dir_var = FIG.peri_posvel_var / f"Target_{target_label}"
        fig_dir_var.mkdir(parents=True, exist_ok=True)

        out_svg_var = fig_dir_var / f"Cond_{br_idx:03d}__peristim_VAR.png"

        rcp.stacked_heatmaps_plus_behv(
            NPRW_var,
            ua_plot_var,
            NPRW_rel_t if NPRW_var.size else None,
            UA_rel_t   if UA_var.size   else None,
            out_svg_var,
            title_kinematics,
            title_NA_var,
            cmap=COLORMAP,
            vmin_nprw=VMIN_NPRW_VAR,
            vmax_nprw=VMAX_NPRW_VAR,
            vmin_ua={
                "M1i+M1s": VMIN_UA_VAR,
                "PMd":     VMIN_UA_VAR,
                "SMA":     0.0,
            },
            vmax_ua={
                "M1i+M1s": VMAX_UA_VAR,
                "PMd":     VMAX_UA_VAR,
                "SMA":     VMAX_SMA_VAR,
            },
            probe=nprw_probe, probe_locs=locs, stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="none",
            beh_rel_time=beh_time_for_both,
            beh_cam0_lines=beh_cam0_pos_med,
            beh_cam1_lines=beh_cam1_pos_med,
            beh_cam0_vel_lines=beh_cam0_vel_med,
            beh_cam1_vel_lines=beh_cam1_vel_med,
            beh_labels=beh_labels,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=overall_title,
        )
        print(f"[peristim] wrote {out_svg_var}")

# ---------- main loop ----------

def main():
    """
    For each of PeriStim/Target_A and PeriStim/Target_B:
      - plot all baseline__*.npz
      - plot all peristim__*.npz
    Figures go into PeriStim/Target_X/figures.
    """
    # Build impedances once per run
    if EXCLUDE_UA_HIGH_Z:
        ua_imp_a, ua_imp_b = _build_impedance_dict_for_target()
    else:
        ua_imp_a, ua_imp_b = {}, {}

    for target_label in ("A", "B"):
        target_dir = PERI_ROOT / f"Target_{target_label}"
        if not target_dir.exists():
            print(f"[peri/baseline-plot] Target_{target_label} not found at {target_dir}, skipping.")
            continue

        # ---------- Baselines ----------
        base_npz_files = sorted(target_dir.glob("baseline__*.npz"))
        if base_npz_files:
            print(f"[peri/baseline-plot] Target_{target_label}: {len(base_npz_files)} baseline NPZ files.")
            for npz_path in base_npz_files:
                # choose port from filename, not target_label
                port = _port_from_baseline_path(npz_path)
                if port == "A":
                    ua_imp = ua_imp_a
                elif port == "B":
                    ua_imp = ua_imp_b
                else:
                    ua_imp = {}
                    print(f"[warn] Could not infer UA port from baseline file: {npz_path.name}")
                _plot_one_baseline_npz(npz_path, target_label, ua_imp)
        else:
            print(f"[peri/baseline-plot] Target_{target_label}: no baseline__*.npz found.")

        # ---------- Peri-stim ----------
        peri_npz_files = sorted(target_dir.glob("peristim__*.npz"))
        if peri_npz_files:
            print(f"[peri/baseline-plot] Target_{target_label}: {len(peri_npz_files)} peristim NPZ files.")
            for npz_path in peri_npz_files:
                _plot_one_peristim_npz(npz_path, target_label, ua_imp_a, ua_imp_b)
        else:
            print(f"[peri/baseline-plot] Target_{target_label}: no peristim__*.npz found.")


if __name__ == "__main__":
    main()
