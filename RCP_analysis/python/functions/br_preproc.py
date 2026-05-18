from pathlib import Path
from typing import Any
import csv
import numpy as np
import pandas as pd
import spikeinterface.extractors as se

def list_br_sessions(br_root: Path) -> list[Path]:
    """
    List Blackrock sessions under br_root.

    A session can be either:
      - a subdirectory, or
      - a base filename for NSx/NEV pairs in br_root.
    """
    root = br_root
    if not root.exists():
        raise FileNotFoundError(f"Session root not found: {root}")

    # Case 1: sessions are organized as subdirectories
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if subdirs:
        return sorted(subdirs)

    # Case 2: sessions are loose NSx/NEV files in the root
    files = [p for p in root.iterdir() if p.is_file()]
    nsx_nev = [
        p for p in files
        if p.suffix.lower().startswith(".ns") or p.suffix.lower() == ".nev"
    ]
    if not nsx_nev:
        raise FileNotFoundError(f"No NSx/NEV files found under {root}.")
    bases = sorted({p.stem for p in nsx_nev})
    return [root / b for b in bases]

def ua_excel_path(repo_root: Path, probes_cfg: dict | None) -> np.ndarray | None:
    ua_cfg = (probes_cfg).get("UA")
    rel = ua_cfg.get("mapping_mat_rel")
    if not rel:
        return None
    p = (repo_root / rel) if not str(rel).startswith("/") else Path(rel)
    return p if p.exists() else None

def extract_br_aux_streams_npz(
    sess,
    aux_dir: Path,
    camera_sync_ch: int,
    triangle_sync_ch: int,
    touchscreen_ch: int,
    hr_ch: int,
    vog_sig_ch: int,
) -> tuple[
    Path | None,
    np.ndarray | None, np.ndarray | None, np.ndarray | None,
    dict[str, Any], dict[str, Any]
]:
    """
    Build and save BR aux streams in a unified format into ONE NPZ file:

      - ns5: selected sync channels (camera, triangle, optional extra) stacked as rows.
      - ns2: all channels stacked as rows, plus name-based extraction of:
            hhpos           -> vog_sig
            vog_sync        -> vog_sync
            touchscreen_syn -> touchscreen_sig
            heart_rate      -> hr_sig

    Returns
    -------
    (out_npz, touchscreen_sig, hr_sig, vog_sig, meta_ns5, meta_ns2)

      meta_ns5: dict with ns5 fields (or {} if ns5 missing)
      meta_ns2: dict with ns2 fields (or {} if ns2 missing)

    Notes
    -----
    meta_ns5/meta_ns2 are intended to be merged into a larger meta dict by the caller.
    """
    aux_dir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {}
    meta_ns5: dict[str, Any] = {}
    meta_ns2: dict[str, Any] = {}
    out_npz: np.ndarray | None

    # ns5
    try:
        rec_ns5 = se.read_blackrock(str(sess), stream_name="nsx5", all_annotations=True)

        wanted = [camera_sync_ch, triangle_sync_ch]
        chan_ids_ns5 = rec_ns5.get_channel_ids().astype(int)
        traces_ns5 = []
        labels_ns5 = []
        ns5_fs_hz = rec_ns5.get_sampling_frequency()

        for ch in wanted:
            if ch in chan_ids_ns5:
                tr = rec_ns5.get_traces(channel_ids=[str(ch)], return_scaled=True).ravel()
                traces_ns5.append(tr)
                labels_ns5.append(ch)
            else:
                print(f"[WARN] ns5: channel {ch} not found.")

        if traces_ns5:
            aux_traces_ns5 = np.stack(traces_ns5, axis=0)
            meta_ns5.update(
                ns5_aux_traces=aux_traces_ns5,
                ns5_session=getattr(sess, "name", str(sess)),
                ns5_stream_name="nsx5",
                ns5_fs_hz=ns5_fs_hz,
                ns5_channel_ids=np.array(labels_ns5, dtype=int),
                fs_camera_sync=ns5_fs_hz,
                fs_triangle=ns5_fs_hz,
                fs_touchscreen=ns5_fs_hz,
            )
            meta.update(meta_ns5)
            print(f"[AUX] prepared ns5 aux (channels {labels_ns5})")
        else:
            print("[WARN] ns5 found but no requested sync channels present; nothing saved for ns5.")

    except FileNotFoundError:
        print("[WARN] ns5 not found; syncs unavailable.")

    # ns2
    vog_sig = touchscreen_sig = hr_sig = None

    try:
        rec_ns2 = se.read_blackrock(str(sess), stream_name="nsx2", all_annotations=True)
        aux_traces_ns2 = rec_ns2.get_traces(return_scaled=True).T  # (n_channels, n_samples)
        ns2_chs = np.asarray(rec_ns2.get_channel_ids())

        # channel_name per channel (no lowercasing / mangling)
        chan_names = []
        for ch_id in ns2_chs:
            nm = rec_ns2.get_channel_property(ch_id, "channel_name")
            if isinstance(nm, bytes):
                nm = nm.decode("utf-8", errors="ignore")
            chan_names.append(str(nm))
        chan_names = np.array(chan_names)

        def _grab_by_name(target_name: str, label: str):
            """Return (signal, description_string)."""
            idxs = np.where(chan_names == target_name)[0]
            if idxs.size > 0:
                idx = int(idxs[0])
                sig = aux_traces_ns2[idx]
                desc = f"{label}={target_name}(id {ns2_chs[idx]})"
                return sig, desc
            else:
                return None, f"{label}=None"

        # Name-based extraction
        vog_sig, desc_vog            = _grab_by_name("hhpos",           "vog_sig")
        vog_sync, desc_vog_sync      = _grab_by_name("vog_sync",        "vog_sync")
        touchscreen_sig, desc_touch  = _grab_by_name("touchscreen_syn", "touchscreen_sig")
        hr_sig, desc_hr              = _grab_by_name("heart_rate",      "hr_sig")
        ns2_fs_hz = float(rec_ns2.get_sampling_frequency())

        meta_ns2.update(
            ns2_aux_traces=aux_traces_ns2,
            ns2_session=getattr(sess, "name", str(sess)),
            ns2_stream_name="nsx2",
            ns2_fs_hz=ns2_fs_hz,
            ns2_channel_ids=ns2_chs.astype(int),
            ns2_channel_names=chan_names,
        )

        # per-signal sampling rates (signals are extracted from ns2)
        meta_ns2.update(
            fs_ts=ns2_fs_hz,
            fs_hr=ns2_fs_hz,
            fs_vog=ns2_fs_hz,
        )

        if vog_sync is not None:
            meta_ns2["ns2_vog_sync_trace"] = vog_sync

        meta.update(meta_ns2)

        # single summary line
        print("[AUX] ns2 signals: " + ", ".join([desc_vog, desc_vog_sync, desc_touch, desc_hr]))
        print(f"[AUX] prepared ns2 digi (n_channels={aux_traces_ns2.shape[0]})")

    except FileNotFoundError:
        print("[WARN] ns2 not found; no digital channels.")

    if meta:
        out_npz = aux_dir / f"{getattr(sess, 'name', Path(sess).name)}__BR_aux_data.npz"
        np.savez_compressed(out_npz, **meta)
        print(f"[AUX] saved combined BR aux streams to {out_npz}")
    else:
        print("[AUX] No aux streams found; nothing saved.")

    return touchscreen_sig, hr_sig, vog_sig, meta_ns5, meta_ns2

# Mapping
def apply_ua_mapping_with_regions(
    recording,
    mapped_nsp: np.ndarray,
    br_idx: int,
    meta_csv: Path,
    port: str | None = None,
):
    """
    Apply UA mapping by renaming channels and build UA-related per-row arrays.

    Assumptions:
    - metadata CSV has columns: BR_File, UA_port
    - UA_port is 'A' or 'B'
    - recording.get_channel_ids() -> local NSP ids 1..128 (per port)
    - mapped_nsp uses 1..128 for Port A, 129..256 for Port B

    Parameters
    ----------
    port : str or None
        If 'A' or 'B', overrides the UA_port from the metadata CSV.
        If None, UA_port is read from meta_csv (BR_File row).

    Returns
    -------
    renamed : recording with renamed channel_ids
    idx_rows : np.ndarray (n_electrodes,), electrode -> recording row index (or -1)
    ua_elec : np.ndarray (n_channels,), row -> UA electrode number (or -1)
    ua_nsp  : np.ndarray (n_channels,), row -> NSP id (or -1)
    ua_region : np.ndarray (n_channels,), row -> region index {-1,0,1,2,3}
    ua_region_names   : np.ndarray (4,), ["SMA", "Dorsal premotor", "M1 inferior", "M1 superior"]
    ua_port:
    """
    ch_ids = np.asarray(recording.get_channel_ids(), int)
    n_channels = ch_ids.size

    # ---- decide which UA port to use ----
    ua_port = None

    # 1) Explicit override from function arg, if provided
    if port is not None:
        ua_port = str(port).strip().upper()
        print(f"[MAP] UA port override from argument: {ua_port!r}")
    else:
        # 2) Otherwise, read from metadata CSV
        if not meta_csv.exists():
            print(f"[WARN] metadata CSV not found at {meta_csv}")
        else:
            with meta_csv.open("r", newline="") as f:
                rdr = csv.DictReader(f)
                if not rdr.fieldnames:
                    print(f"[WARN] {meta_csv.name} has no header")
                elif "BR_File" not in rdr.fieldnames or "UA_port" not in rdr.fieldnames:
                    print(f"[WARN] metadata missing BR_File and/or UA_port columns (have: {rdr.fieldnames})")
                else:
                    for row in rdr:
                        try:
                            if int(row["BR_File"]) == br_idx:
                                ua_port = (row["UA_port"] or "").strip().upper() or None
                                break
                        except Exception:
                            continue
        if ua_port is not None:
            print(f"[MAP] UA port from metadata: {ua_port!r}")

    # 3) Fallback if still unknown
    if ua_port not in ("A", "B"):
        print(f"[WARN] UA port unknown or invalid ({ua_port!r}); assuming 'A' (no NSP offset).")
        ua_port = "A"

    # ---- If Port B, shift local 1..128 -> NSP 129..256 ----
    if ua_port == "B":
        ch_ids = ch_ids + 128
        print("[MAP] Applying NSP offset for Port B: local 1..128 -> NSP 129..256")

    # ---- NSP id -> row index ----
    id_to_row = {ch: i for i, ch in enumerate(ch_ids)}
    mapped_nsp_int = mapped_nsp.astype(int, copy=False)

    # electrode -> row index (or -1)
    idx_rows = np.full(mapped_nsp_int.shape, -1)
    for elec_idx, nsp_id in enumerate(mapped_nsp_int):
        idx_rows[elec_idx] = id_to_row.get(int(nsp_id), -1)

    # ---- rename channels to UAe###_NSP### where applicable ----
    new_ids = [str(ch) for ch in ch_ids]
    for elec0, (row, nsp) in enumerate(zip(idx_rows, mapped_nsp_int)):
        if 0 <= row < n_channels:
            elec = elec0 + 1  # 1-based electrode index
            new_ids[row] = f"UAe{elec:03d}_NSP{nsp:03d}"

    renamed = recording.rename_channels(new_ids)
    mapped = np.count_nonzero((idx_rows >= 0) & (idx_rows < n_channels))
    print(
        f"[MAP] renamed {mapped}/{n_channels} rows with UA mapping "
        f"('UAe###_NSP###') using port {ua_port!r}."
    )

    renamed.set_annotation("ua_row_index", idx_rows)

    # ---- build per-row UA arrays ----
    ua_elec = -np.ones(n_channels)
    ua_nsp  = -np.ones(n_channels)
    for elec0, row in enumerate(idx_rows):
        if 0 <= row < n_channels:
            ua_elec[row] = elec0 + 1
            ua_nsp[row]  = int(mapped_nsp_int[elec0])

    # region assignment from electrode number
    def _ua_region_from_elec(e: int) -> int:
        if e <= 0:    return -1
        if e <= 64:   return 0  # SMA
        if e <= 128:  return 1  # Dorsal premotor
        if e <= 192:  return 2  # M1 inferior
        return 3                # M1 superior

    ua_region = np.fromiter(
        (_ua_region_from_elec(int(e)) for e in ua_elec),
        dtype=np.int8,
        count=n_channels,
    )
    ua_region_names = np.array(["SMA", "Dorsal premotor", "M1 inferior", "M1 superior"])

    return (
        renamed,
        idx_rows,
        ua_elec,
        ua_nsp,
        ua_region,
        ua_region_names,
        ua_port,
    )

def load_UA_mapping_from_excel(
    xls_path: Path,
    sheet: str | int = 0,
    n_elec: int | None = None,
) -> dict[str, Any]:
    """
    Use pandas only to load the sheet, then convert columns to numpy
    and do all processing in pure numpy.
    """
    xls_path = Path(xls_path)
    if not xls_path.exists():
        raise FileNotFoundError(f"Excel not found: {xls_path}")

    # Load into DataFrame
    df = pd.read_excel(xls_path, sheet_name=sheet)

    if "NSP ch" not in df.columns or "Elec#" not in df.columns:
        raise ValueError(
            f"Expected columns NSP ch and Elec# not found.\n"
            f"Columns present: {list(df.columns)}"
        )

    # Convert to NumPy
    nsp_np  = df["NSP ch"].to_numpy()
    elec_np = df["Elec#"].to_numpy()

    # ----- Parse both columns with NumPy loops -----
    parsed_nsp = np.char.replace(nsp_np.astype(str), "ch-", "").astype(int)
    parsed_elec = np.array([int(v) if v is not None and not pd.isna(v) else None
                            for v in elec_np], dtype=object)

    # valid mask
    mask = [(parsed_nsp[i] is not None and parsed_elec[i] is not None)
            for i in range(len(parsed_nsp))]

    nsp  = np.array([parsed_nsp[i]  for i in range(len(mask)) if mask[i]])
    elec = np.array([parsed_elec[i] for i in range(len(mask)) if mask[i]])

    # ----- Build mapping -----
    max_elec = int(n_elec or elec.max())
    mapped_nsp = np.zeros(max_elec)
    mapped_nsp[elec - 1] = nsp

    return mapped_nsp

def load_electrode_mapping(csv_path: Path):
    """
    Load electrode mapping from CSV.
    Returns:
        nsp_to_elec: {nsp_id: electrode_id} mapping.
        region_grids: {region: 8x8_array} of electrode IDs.
        elec_to_region: {electrode_id: region} mapping.
    """
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

def get_region_from_group_name(grp_name: str) -> str:
    """Extracts region name from group string like 'M1s (n=45)' -> 'M1s'."""
    return grp_name.split(" (")[0].strip()

def get_region_grid(region_name: str, utah_elec_grids: dict):
    """Gets the 8x8 grid for a given region."""
    aliases = {"M1 Inf": "M1i", "M1 Sup": "M1s"}
    target = aliases.get(region_name, region_name)
    return utah_elec_grids.get(target, None)

def build_elec_to_data_idx(ua_ids_1based, nsp_to_elec, ua_port='A'):
    """
    Builds electrode_id -> channel index mapping.
    
    For Port A: data channels 0-127 correspond to NSP 1-128
    For Port B: data channels 0-127 correspond to NSP 129-256
    """
    elec_to_idx = {}
    
    # Determine NSP offset based on port
    nsp_offset = 0 if ua_port == 'A' else 128
    
    # For each data channel index (0-127), determine which electrode it maps to
    n_channels = len(ua_ids_1based) if ua_ids_1based is not None else 128
    
    for ch_idx in range(n_channels):
        nsp_id = ch_idx + 1 + nsp_offset  # Channel 0 -> NSP 1 (Port A) or NSP 129 (Port B)
        elec_id = nsp_to_elec.get(nsp_id, -1)
        if elec_id > 0:
            elec_to_idx[elec_id] = ch_idx
    
    return elec_to_idx

__all__ = [
    "list_br_sessions", "ua_excel_path", "extract_br_aux_streams_npz", "apply_ua_mapping_with_regions", 
    "load_UA_mapping_from_excel", "load_electrode_mapping", "get_region_from_group_name", 
    "get_region_grid", "build_elec_to_data_idx"
]