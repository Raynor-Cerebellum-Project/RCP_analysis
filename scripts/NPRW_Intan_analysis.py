from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import gc
import numpy as np

# SpikeInterface
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
from spikeinterface.core import concatenate_recordings
from spikeinterface.exporters import export_to_phy

# Project config
from RCP_analysis import load_experiment_params, resolve_data_root

# Intan helpers
from RCP_analysis import (
    load_stim_geometry,
    make_probe_from_geom,
    read_intan_recording,
    local_cm_reference,
    save_recording,
    list_intan_sessions,
)
from pathlib import Path
from typing import Tuple

# Package API
from RCP_analysis import (
    load_experiment_params, resolve_data_root, resolve_output_root,
    resolve_probe_geom_path, resolve_session_intan_dir,
    resolve_probe_geom_path,
)

# ==============================
# Config
# ==============================
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE = resolve_output_root(PARAMS)
OUT_BASE.mkdir(parents=True, exist_ok=True)
DATA_ROOT = resolve_data_root(PARAMS)

# --- Intan root ---
INTAN_ROOT = (
    Path(PARAMS.intan_root).resolve()
    if getattr(PARAMS, "intan_root", None) and str(PARAMS.intan_root).startswith("/")
    else (DATA_ROOT / PARAMS.intan_root_rel).resolve()
    if getattr(PARAMS, "intan_root_rel", None)
    else None
)
if INTAN_ROOT is None:
    raise ValueError("No Intan root specified. Set 'intan_root_rel' or 'intan_root' in params.yaml.")

# --- Geometry path ---
GEOM_PATH = (
    Path(PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
    else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None)
    else resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
)

# === Intan stream name ===
INTAN_STREAM = getattr(PARAMS, "neural_data_stream", "RHS2000 amplifier channel")
STIM_STREAM = getattr(PARAMS, "stim_data_stream", "RHS2000 amplifier channel")

# === Local reference annulus (µm) ===
probe_cfg = (PARAMS.probes or {}).get("NPRW", {})
inner = float(probe_cfg.get("local_radius_inner", 30.0))
outer = float(probe_cfg.get("local_radius_outer", 150.0))
ANNULUS: Tuple[float, float] = (inner, outer)

global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)



from pathlib import Path
from neo.rawio.intanrawio import IntanRawIO

def inspect_intan_streams(sess_folder: Path):
    """
    Print available Intan streams and channels in a session folder.

    Shows:
      - signal streams (names/ids) and sampling rates
      - signal channels in each stream
      - event (digital) channels – likely where stim/sync lives
    """
    r = IntanRawIO(dirname=str(sess_folder))
    r.parse_header()

    # Streams (continuous signals)
    streams = r.header['signal_streams']   # array of (stream_id, stream_name)
    chans   = r.header['signal_channels']  # array with fields: name, id, stream_id, dtype, sampling_rate, units

    print(f"\n=== STREAMS in {sess_folder.name} ===")
    for sid, sname in streams:
        # channels belonging to this stream
        cs = [c for c in chans if c['stream_id'] == sid]
        srates = {float(c['sampling_rate']) for c in cs}
        sr_str = ", ".join(sorted({f"{sr:g} Hz" for sr in srates}))
        print(f"- stream_id='{sid}'  name='{sname}'  (samplerate(s): {sr_str}, n_channels={len(cs)})")
        # show a few channel names
        preview = ", ".join(c['name'] for c in cs[:6])
        if len(cs) > 6:
            preview += ", ..."
        print(f"    channels: {preview}")

    # Event/digital channels (TTL lines, buttons, stim markers, etc.)
    ev = r.header.get('event_channels', None)
    if ev is not None and len(ev):
        print("\n=== EVENT (DIGITAL) CHANNELS ===")
        for i, e in enumerate(ev):
            # e has fields: name, id, dtype
            print(f"- idx={i:02d} name='{e['name']}' id='{e['id']}' dtype={e['dtype']}")
    else:
        print("\n(no event channels found)")

    print()  # newline for readability




# ==============================
# Pipeline
# ==============================
def main(use_br: bool = False, use_intan: bool = True, limit_sessions: Optional[int] = None):
    # 1) Load geometry & mapping
    geom = load_stim_geometry(GEOM_PATH)
    probe = make_probe_from_geom(geom, radius_um=5.0)

    # 2) Find sessions & load each Intan folder
    sess_folders = list_intan_sessions(INTAN_ROOT)
    if limit_sessions:
        sess_folders = sess_folders[:limit_sessions]
    print(f"Found Intan sessions: {len(sess_folders)}")

    preproc_paths: list[Path] = []
    checkpoint_out = OUT_BASE / "checkpoints/NPRW"
    checkpoint_out.mkdir(parents=True, exist_ok=True)

    for sess in sess_folders:
        print(f"=== Session: {sess.name} ===")

        # Load Intan
        rec = read_intan_recording(sess, stream_name=INTAN_STREAM)
        rec = rec.set_probe(probe, in_place=False)

        # 3) Local CMR (inner/outer radius)
        rec_ref = local_cm_reference(rec, freq_min=float(PARAMS.highpass_hz), inner_outer_radius_um=ANNULUS)

        # Ensure channel locations exist (Kilosort4 requires geometry)
        try:
            _ = rec_ref.get_channel_locations()
        except Exception:
            n_ch = rec_ref.get_num_channels()
            locs = np.column_stack([np.arange(n_ch, dtype=float), np.zeros(n_ch, dtype=float)])
            rec_ref.set_channel_locations(locs)

        # Persist preprocessed session
        out_dir = checkpoint_out / f"pp_local_{int(ANNULUS[0])}_{int(ANNULUS[1])}__{sess.name}"
        save_recording(rec_ref, out_dir)
        print(f"[{sess.name}] saved preprocessed -> {out_dir}")

        del rec, rec_ref
        gc.collect()

        preproc_paths.append(out_dir)

    if not preproc_paths:
        raise RuntimeError("No Intan sessions processed; nothing to concatenate.")

    # 4) Concatenate
    print("Concatenating preprocessed sessions...")
    recs = []
    for p in preproc_paths:
        try:
            r = si.load(p)
        except Exception:
            r = si.load_extractor(p / "si_folder.json")
        recs.append(r)
    rec_concat = concatenate_recordings(recs)
    gc.collect()

    # 5) Kilosort 4
    ks4_out = OUT_BASE / "kilosort4"
    sorting_ks4 = ss.run_sorter(
        "kilosort4",
        recording=rec_concat,
        folder=str(ks4_out),
        remove_existing_folder=True,
        verbose=True,
    )

    # 6) Export to Phy
    sa_folder = OUT_BASE / "sorting_ks4_analyzer"
    phy_folder = OUT_BASE / "phy_ks4"

    sa = si.create_sorting_analyzer(
        sorting=sorting_ks4,
        recording=rec_concat,
        folder=sa_folder,
        overwrite=True,
        sparse=True,
    )

    # Phy needs at least waveforms/templates/PCs; random_spikes makes it deterministic
    sa.compute("random_spikes", method="uniform", max_spikes_per_unit=1000, seed=0)
    sa.compute("waveforms", ms_before=1.0, ms_after=2.0, max_spikes_per_unit=1000,
               n_jobs=int(PARAMS.parallel_jobs), chunk_duration=str(PARAMS.chunk), progress_bar=True)
    sa.compute("templates")
    sa.compute("principal_components", n_components=5, mode="by_channel_local",
               n_jobs=int(PARAMS.parallel_jobs), chunk_duration=str(PARAMS.chunk), progress_bar=True)
    sa.compute("spike_amplitudes")

    export_to_phy(sa, output_folder=phy_folder, copy_binary=True, remove_if_exists=True)
    print(f"Phy export ready: {phy_folder}")


if __name__ == "__main__":
    main(limit_sessions=None)
