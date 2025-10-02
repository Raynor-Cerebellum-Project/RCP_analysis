# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional
import gc
import numpy as np
import io, codecs, re
import csv

# SpikeInterface
import spikeinterface as si
import spikeinterface.preprocessing as spre
from sklearn.decomposition import PCA

from RCP_analysis import (
    load_experiment_params, resolve_data_root, resolve_output_root,
    # BR/UA helpers
    list_br_sessions, ua_excel_path, load_ns6_spikes,
    load_UA_mapping_from_excel, apply_ua_mapping_properties,
    build_blackrock_bundle, save_UA_bundle_npz, threshold_mua_rates, # Not sure why this isn't color coded
    load_stim_detection,
)

def _session_from_rates_path(p: Path) -> str:
    # rates__NRR_RW001_250915_141052__bin1ms_sigma25ms.npz -> NRR_RW001_250915_141052
    stem = p.stem
    body = stem[len("rates__"):]
    return body.split("__bin", 1)[0]

def _parse_intan_session_dtkey(session: str) -> int:
    # Sort sessions by YYMMDD_HHMMSS at end of name if present
    m = re.search(r"(\d{6})_(\d{6})$", session)
    if not m:
        return int("9"*12)
    return int(m.group(1) + m.group(2))

def _build_session_index_map(nprw_ckpt_root: Path) -> tuple[dict[str,int], dict[int,str]]:
    """Scan NPRW rates to build Intan session ↔ Intan_File index mapping."""
    rate_files = sorted(nprw_ckpt_root.rglob("rates__*.npz"))
    sessions = sorted({_session_from_rates_path(p) for p in rate_files},
                      key=_parse_intan_session_dtkey)
    sess_to_intan = {s: i+1 for i, s in enumerate(sessions)}
    intan_to_sess = {i+1: s for i, s in enumerate(sessions)}
    if not sessions:
        raise RuntimeError(f"No NPRW rate files under {nprw_ckpt_root}")
    return sess_to_intan, intan_to_sess

def _read_intan_to_br_map(csv_path: Path) -> dict[int, int]:
    """
    Robust CSV reader. Returns {Intan_File index -> BR_File index}.
    Accepts headers like: Intan_File, BR_File (case/punct in any form).
    """
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
    raw = csv_path.read_bytes()
    try_order = []
    if raw.startswith(codecs.BOM_UTF8):
        try_order = ["utf-8-sig"]
    elif raw.startswith(codecs.BOM_UTF16_LE):
        try_order = ["utf-16-le"]
    elif raw.startswith(codecs.BOM_UTF16_BE):
        try_order = ["utf-16-be"]
    try_order += ["utf-8", "cp1252", "latin1", "utf-16"]
    text = None
    for enc in try_order:
        try:
            text = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("latin1", errors="replace")

    rdr = csv.DictReader(io.StringIO(text))
    if not rdr.fieldnames:
        raise ValueError("CSV has no header row")
    fmap = {norm(k): k for k in rdr.fieldnames if k}

    def pick(*names):
        for n in names:
            if n in fmap: return fmap[n]
        raise KeyError(f"CSV missing one of: {names}")

    col_intan = pick("intan_file","intanfile","intan","intanindex","intanfileindex")
    col_br    = pick("br_file","brfile","br","brindex","brfileindex")

    out = {}
    for row in rdr:
        try:
            out[int(str(row[col_intan]).strip())] = int(str(row[col_br]).strip())
        except Exception:
            pass
    if not out:
        raise ValueError("No (Intan, BR) rows parsed")
    return out

def _resolve_intan_session_for_br_idx(
    out_base: Path,
    br_idx: int,
    nprw_ckpt_root: Path
) -> Optional[str]:
    """
    Use Metadata/NRR_RW001_metadata.csv and NPRW rates to map BR index -> Intan session name.
    """
    meta_csv = out_base.parent / "Metadata" / "NRR_RW001_metadata.csv"
    if not meta_csv.exists():
        print(f"[WARN] mapping CSV not found: {meta_csv}")
        return None

    # 1) {Intan_File -> BR_File}
    intan_to_br = _read_intan_to_br_map(meta_csv)

    # 2) invert to {BR_File -> Intan_File}
    br_to_intan = {b: i for i, b in intan_to_br.items()}

    if br_idx not in br_to_intan:
        print(f"[WARN] BR {br_idx:03d} not in mapping CSV.")
        return None

    intan_idx = br_to_intan[br_idx]

    # 3) Build {Intan_File index -> session string} from NPRW rates
    _, intan_idx_to_sess = _build_session_index_map(nprw_ckpt_root)

    sess = intan_idx_to_sess.get(intan_idx)
    if not sess:
        print(f"[WARN] Intan_File {intan_idx} not found among NPRW rates.")
        return None
    return sess

def _stim_npz_for_br_idx(out_base: Path, br_idx: int, nprw_ckpt_root: Path) -> Optional[Path]:
    sess = _resolve_intan_session_for_br_idx(out_base, br_idx, nprw_ckpt_root)
    if not sess:
        return None
    bundles_root = out_base / "bundles" / "NPRW"
    return bundles_root / f"{sess}_Intan_bundle" / "stim_stream.npz"

def _br_idx_from_name(name: str) -> Optional[int]:
    # Prefer the trailing 3 digits after the last underscore
    m = re.search(r'_(\d{3})$', name)
    if m:
        return int(m.group(1))
    # Fallback: take the last 3-digit group if the name has a suffix
    hits = re.findall(r'_(\d{3})(?=_)', name)
    return int(hits[-1]) if hits else None

def _load_shift_row_by_br_idx(out_base: Path, br_idx: int) -> Optional[dict]:
    """
    Read figures/align_BR_to_Intan/br_to_intan_shifts.csv and return the row for this br_idx.
    Expected columns include: session (Intan), br_idx, anchor_ms, fs_intan, (optionally anchor_sample, etc.)
    """
    shifts_csv = out_base / "figures" / "align_BR_to_Intan" / "br_to_intan_shifts.csv"
    if not shifts_csv.exists():
        return None
    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                if int(row.get("br_idx", "-1")) == int(br_idx):
                    return row
            except Exception:
                continue
    return None

def _anchor_from_shifts_row(row: dict) -> tuple[int, float]:
    """
    From a shifts row, return (anchor_sample_intan, fs_intan).
    If anchor_sample is missing, derive from anchor_ms.
    """
    fs_intan = float(row.get("fs_intan", 30000.0))
    if row.get("anchor_sample", "") != "":
        return int(row["anchor_sample"]), fs_intan
    if row.get("anchor_ms", "") != "":
        anchor_ms = float(row["anchor_ms"])
        return int(round(anchor_ms * 1e-3 * fs_intan)), fs_intan
    return 0, fs_intan

def _stim_npz_from_shifts_row(out_base: Path, row: dict) -> Path:
    """
    Build the path to the Intan stim detection NPZ from the shifts row.
    """
    intan_session = row["session"]  # Intan session name from compute_br_to_intan_shifts.py
    bundles_root = out_base / "bundles" / "NPRW"
    return bundles_root / f"{intan_session}_Intan_bundle" / "stim_stream.npz"

def load_anchor_for_session(out_base: Path, session: str) -> tuple[int, float]:
    """
    Return (anchor_sample_intan, fs_intan) from figures/align_BR_to_Intan/br_to_intan_shifts.csv.
    Falls back to deriving anchor_sample from anchor_ms if needed.
    """
    shifts_csv = out_base / "figures" / "align_BR_to_Intan" / "br_to_intan_shifts.csv"
    if not shifts_csv.exists():
        return 0, 30000.0  # safe fallback

    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get("session") == session:
                fs_intan = float(row.get("fs_intan", 30000.0))
                if row.get("anchor_sample", "") != "":
                    return int(row["anchor_sample"]), fs_intan
                if row.get("anchor_ms", "") != "":
                    anchor_ms = float(row["anchor_ms"])
                    return int(round(anchor_ms * 1e-3 * fs_intan)), fs_intan
                break

    return 0, 30000.0  # fallback if no row

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)

OUT_BASE = resolve_output_root(PARAMS)
OUT_BASE.mkdir(parents=True, exist_ok=True)

global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)

def main(use_br: bool = True, use_intan: bool = False, limit_sessions: Optional[int] = None):
    data_root = resolve_data_root(PARAMS)
    session_folders = list_br_sessions(data_root, PARAMS.blackrock_rel)
    if limit_sessions:
        session_folders = session_folders[:limit_sessions]
    print("Found session folders:", len(session_folders))

    RATES = PARAMS.UA_rate_est or {}
    BIN_MS     = float(RATES.get("bin_ms", 1.0))
    SIGMA_MS   = float(RATES.get("sigma_ms", 50.0))
    THRESH     = float(RATES.get("detect_threshold", 4.5))
    PEAK_SIGN  = str(RATES.get("peak_sign", "neg"))
        
    xls = ua_excel_path(REPO_ROOT, PARAMS.probes)
    ua_map = load_UA_mapping_from_excel(xls) if xls else None
    if ua_map is None:
        raise RuntimeError("UA mapping required for mapping on NS6.")

    bundles_out = OUT_BASE / "bundles" / "UA"
    checkpoint_out = OUT_BASE / "checkpoints" / "UA"
    nprw_ckpt_root = OUT_BASE / "checkpoints" / "NPRW"
    checkpoint_out.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # --- per session: save non-spike bundle + preproc ns6
    for sess in session_folders:
        print(f"=== Session: {sess.name} ===")
    
        bundle = build_blackrock_bundle(sess)
        save_UA_bundle_npz(sess.name, bundle, bundles_out)

        rec_ns6 = load_ns6_spikes(sess)
        apply_ua_mapping_properties(rec_ns6, ua_map["mapped_nsp"])  # metadata only

        rec_hp  = spre.highpass_filter(rec_ns6, freq_min=float(PARAMS.highpass_hz))
        rec_ref = spre.common_reference(rec_hp, reference="global", operator="median")

        try:
            _ = rec_ref.get_channel_locations()
        except Exception:
            n_ch = rec_ref.get_num_channels()
            locs = np.column_stack([np.arange(n_ch, dtype=float), np.zeros(n_ch, dtype=float)])
            rec_ref.set_channel_locations(locs)
        
        # --- Resolve br_idx for this UA folder from its name ---
        br_idx = _br_idx_from_name(sess.name)

        # Default: no artifact windows
        block_bounds = np.empty((0, 2), dtype=int)
        shifts_row = None

        if br_idx is None:
            print(f"[WARN] Could not parse BR index from session folder '{sess.name}'. Skipping artifact removal.")
        else:
            # 1) find Intan session via metadata + NPRW rates
            stim_npz_path = _stim_npz_for_br_idx(OUT_BASE, br_idx, nprw_ckpt_root)
            # 2) and fetch the shift row for the same BR index
            shifts_row = _load_shift_row_by_br_idx(OUT_BASE, br_idx)

            if not stim_npz_path or not stim_npz_path.exists():
                print(f"[WARN] stim_stream.npz not found for BR {br_idx:03d} (looked at {stim_npz_path}).")
            else:
                # show the resolved Intan session name (parent of stim file)
                intan_session = stim_npz_path.parent.name.replace("_Intan_bundle", "")
                print(f"[map] BR {br_idx:03d} -> Intan session '{intan_session}'")

                stim = load_stim_detection(stim_npz_path)
                block_bounds = np.asarray(stim.get("block_bounds_samples", []), dtype=int)

        # default to cleaned=ref unless we actually remove
        rec_artif_removed = rec_ref
        fs_ua = float(rec_ref.get_sampling_frequency())

        if block_bounds.size and shifts_row is not None:
            # Anchor from the SAME shifts row (important!)
            anchor_samp_intan, fs_intan = _anchor_from_shifts_row(shifts_row)

            starts_intan = block_bounds[:, 0].astype(np.int64)
            ends_intan   = block_bounds[:, 1].astype(np.int64)

            # shift+scale into UA sample index space
            scale = fs_ua / fs_intan
            starts_ua = np.round((starts_intan - anchor_samp_intan) * scale).astype(np.int64)
            ends_ua   = np.round((ends_intan   - anchor_samp_intan) * scale).astype(np.int64)

            # validity + clipping
            n_total = rec_ref.get_num_frames() if hasattr(rec_ref, "get_num_frames") else rec_ref.get_num_samples()
            ends_ua = np.minimum(ends_ua, n_total)
            valid = (ends_ua > starts_ua) & (starts_ua >= 0) & (ends_ua <= n_total)
            starts_ua = starts_ua[valid]
            ends_ua   = ends_ua[valid]

            if starts_ua.size:
                ms_before = 5.0
                tail_ms   = 5.0
                dur_ms    = (ends_ua - starts_ua) * 1000.0 / fs_ua
                ms_after  = float(dur_ms.max() + tail_ms)

                rec_artif_removed = spre.remove_artifacts(
                    rec_ref,
                    list_triggers=[starts_ua.tolist()],
                    ms_before=ms_before,
                    ms_after=ms_after,     # one window long enough for all spans
                    mode="zeros",          # or "linear"
                )
            else:
                print("[WARN] all artifact intervals invalid after shift; skipping artifact removal.")
        else:
            if not block_bounds.size:
                print("[WARN] no block spans found; skipping artifact removal.")
            elif shifts_row is None:
                print("[WARN] no shift row found for this BR index; skipping artifact removal.")

        # save the cleaned preprocessed recording
        out_geom = checkpoint_out / f"pp_global__{sess.name}__NS6"
        out_geom.mkdir(parents=True, exist_ok=True)
        rec_artif_removed.save(folder=out_geom, overwrite=True)
        print(f"[{sess.name}] (ns6) saved preprocessed -> {out_geom}")

        # compute rates FROM THE CLEANED RECORDING
        rate_hz, t_cat_ms, counts_cat, _, _ = threshold_mua_rates(
            rec_artif_removed,
            detect_threshold=THRESH,
            peak_sign=PEAK_SIGN,
            bin_ms=BIN_MS,
            sigma_ms=SIGMA_MS,
            n_jobs=PARAMS.parallel_jobs,
        )

        # rate_hz is (n_channels, n_bins) -> transpose to (n_bins, n_channels)
        X = rate_hz.T

        # PCA for visualization
        pca = PCA(n_components=5, random_state=0)
        pcs = pca.fit_transform(X)            # shape: (n_bins, 5)
        explained_var = pca.explained_variance_ratio_  # shape: (5,)

        # Transpose back to (n_components, n_bins) for consistency
        pcs_T = pcs.T.astype(np.float32)
        
        out_npz = checkpoint_out / f"rates__{sess.name}__bin{int(BIN_MS)}ms_sigma{int(SIGMA_MS)}ms.npz"
        np.savez_compressed(
            out_npz,
            rate_hz=rate_hz.astype(np.float32),
            t_ms=t_cat_ms.astype(np.float32),
            counts=counts_cat.astype(np.uint16),
            pcs=pcs_T,                                  # (5, n_bins)
            explained_var=explained_var.astype(np.float32),
            meta=dict(
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                bin_ms=BIN_MS,
                sigma_ms=SIGMA_MS,
                fs=float(rec_artif_removed.get_sampling_frequency()),
                n_channels=int(rec_artif_removed.get_num_channels()),
                session=str(sess.name),
            ),
        )
        print(f"[{sess.name}] saved rate matrix + PCA -> {out_npz}")

        # cleanup to keep memory stable on long batches
        del bundle, rec_ns6, rec_hp, rec_ref, rec_artif_removed, rate_hz, t_cat_ms, counts_cat
        gc.collect()

        saved_paths.append(out_geom)

    # if not saved_paths:
    #     raise RuntimeError("No sessions processed; nothing to concatenate.")
    
    # # --- concat + sorting (per-channel, no geometry)
    # print("Concatenating preprocessed sessions...")
    # recs_for_concat = []
    # for p in saved_paths:
    #     try:
    #         r = si.load(p)
    #     except Exception:
    #         r = si.load_extractor(p / "si_folder.json")
    #     recs_for_concat.append(r)

    # rec_concat = concatenate_recordings(recs_for_concat)
    # gc.collect()

    # sorting_ms5 = sorters.run_sorter(
    #     "mountainsort5",
    #     recording=rec_concat,
    #     folder=str(OUT_BASE / "mountainsort5"),
    #     remove_existing_folder=True,
    #     verbose=True,
    #     scheme="2",
    #     # scheme1_detect_channel_radius=1,
    #     detect_threshold=6,
    #     npca_per_channel=3,
    #     filter=False, whiten=True,
    #     delete_temporary_recording=True, progress_bar=True,
    # )

    # sa_folder = OUT_BASE / "sorting_ms5_analyzer"
    # phy_folder = OUT_BASE / "phy_ms5"

    # sa = create_sorting_analyzer(
    #     sorting=sorting_ms5,
    #     recording=rec_concat,
    #     folder=sa_folder,
    #     overwrite=True,
    #     sparse=False,
    # )
    # sa.compute("random_spikes", method="uniform", max_spikes_per_unit=1000, seed=0)
    # sa.compute("waveforms", ms_before=1.0, ms_after=2.0, progress_bar=True)
    # sa.compute("templates")
    # sa.compute("principal_components", n_components=3, mode="by_channel_global", progress_bar=True)
    # sa.compute("spike_amplitudes")

    # export_to_phy(sa, output_folder=phy_folder, copy_binary=True, remove_if_exists=True)
    
    # print(f"Phy export ready: {phy_folder}")

if __name__ == "__main__":
    main(use_intan=False, limit_sessions=None)
