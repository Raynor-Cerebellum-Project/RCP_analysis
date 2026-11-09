"""
sort_nprw_kilosort4.py
======================

Full Session NPRW Preprocessing, Stim Artifact Blanking, Multi-Condition Concatenation,
DREDge Motion Estimation, and Kilosort4 Spike Sorting for the Intan 128-Channel Probe.

Key Pipeline Steps:
  1. Session Discovery & Dynamic Trial Validation:
     - Scan all recording blocks in session metadata (Control, Stim, At-Rest).
     - Verify valid trial count (n_trials > 0); DROP / IGNORE any empty/zero-trial conditions
       (e.g. BR 02 in NRR_RW022).
  2. NPRW Geometry Mapping & Stim Artifact Blanking (matching fastigial_sorting.ipynb):
     - Load 128-ch NPRW geometry and mapping from config/probes.
     - Read raw split Intan recordings, convert unsigned to signed, bandpass filter (300-10000 Hz, float32).
     - Apply local median common reference (CMR) with radii (30, 150) um.
     - Extract stim triggers and blank artifact windows [-20 ms, stim_dur + 20 ms] around pulses/trains.
     - Cache blanked recordings per session.
  3. Multi-Condition Concatenation & Motion Correction:
     - Concatenate all valid blanked sessions into rec_concat.
     - Record session_boundaries and sess_names in session_index.pkl.
     - Estimate motion using DREDge preset (spre.compute_motion) and interpolate motion.
     - Save motion-corrected recording (int16).
  4. Kilosort4 Spike Sorting & Automated Curation:
     - Run Kilosort4 with internal drift correction disabled (do_correction=False).
     - Automated curation: remove duplicate spikes, remove excess spikes, remove redundant units.
     - Compute SortingAnalyzer with quality metrics (SNR, firing rate, refractory period violations).
     - Export to Phy for manual visualization if desired.
  5. Save Session Checkpoints:
     - Split continuous spike trains by session boundary and save spike_trains.npz / .mat
       per session under checkpoints/spikesorting/by_session/<sess_name>/NPRW/.
"""

from __future__ import annotations

import gc
import os
import sys
import shutil
import pickle
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from probeinterface import Probe

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.curation as scur
from spikeinterface.curation import compute_merge_unit_groups
from spikeinterface.sortingcomponents.motion import interpolate_motion
from spikeinterface.exporters import export_to_phy

# Add analysis repository root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import RCP_analysis as rcp

try:
    from RCP_analysis.python.functions.config_loading import (
        PARAMS,
        OUT_BASE,
        INTAN_ROOT,
        METADATA_CSV,
        NPRW_AUX_DATA,
        NPRW_CKPT_ROOT,
        ALIGNED_CKPT_ROOT,
        PERI_ROOT,
        SESSION_LOC,
        REPO_ROOT,
    )
    CONFIG_LOADED = True
except Exception as _e:
    print(f"[sort_nprw] Warning: config_loading failed: {_e}")
    CONFIG_LOADED = False
    PARAMS = None
    OUT_BASE = None
    INTAN_ROOT = None
    METADATA_CSV = None
    NPRW_AUX_DATA = None
    NPRW_CKPT_ROOT = None
    ALIGNED_CKPT_ROOT = None
    PERI_ROOT = None
    SESSION_LOC = None
    REPO_ROOT = _REPO_ROOT


# =============================================================================
# Execution Mode Toggles & Quality Metric Defaults
# =============================================================================
# Set POSTPROCESS_ONLY = True to skip motion correction & Kilosort4 and ONLY run
# Quality Metric filtering & SLAy auto-merging on the existing analyzer_pre_slay!
POSTPROCESS_ONLY = True

# Quality metric filtering thresholds (used in postprocessing & CLI defaults)
MIN_FIRING_RATE: float = 0.1       # Minimum firing rate (Hz)
MIN_SNR: float = 2.5               # Minimum SNR (signal-to-noise ratio)
MAX_RP_CONTAMINATION: float = 0.3  # Maximum refractory period contamination (0.3 = 30%)


def _log(msg: str) -> None:
    print(f"[sort_nprw] {msg}", flush=True)


def blank_session(
    rec: si.BaseRecording,
    stim_arrays: Dict[str, Any],
    fs: float,
    ms_before: float = 20.0,
    tail_ms: float = 20.0,
) -> si.BaseRecording:
    """
    Zero out stim artifact windows using this condition's local trigger indices.
    Matches fastigial_sorting.ipynb and NPRW_Intan_analysis_mf.py.
    """
    if not stim_arrays:
        return rec

    n_total = rec.get_num_samples()

    # 1. Check block bounds (multi-pulse trains)
    block_bounds = stim_arrays.get("block_bounds_samples")
    if block_bounds is not None and len(block_bounds) > 0:
        starts = np.asarray(block_bounds)[:, 0]
        ends = np.asarray(block_bounds)[:, 1]
        valid = (ends > starts) & (starts >= 0) & (starts < n_total)
        starts = starts[valid]
        ends = ends[valid]
        if starts.size > 0:
            dur_ms = (ends - starts) * 1000.0 / fs
            ms_after = float(dur_ms.max() + tail_ms)
            _log(f"Blanking {starts.size} stim blocks ([-{ms_before:.1f}, +{ms_after:.1f}] ms)")
            return si.preprocessing.remove_artifacts(
                rec,
                list_triggers=starts.tolist(),
                ms_before=ms_before,
                ms_after=ms_after,
                mode="zeros",
            )

    # 2. Check trigger pairs (single pulses or continuous / sig gen)
    trigger_pairs = stim_arrays.get("trigger_pairs")
    if trigger_pairs is not None and len(trigger_pairs) > 0:
        pulse_onsets = np.asarray(trigger_pairs)[:, 0]
        valid = (pulse_onsets >= 0) & (pulse_onsets < n_total)
        pulse_onsets = pulse_onsets[valid]
        if pulse_onsets.size > 0:
            _log(f"Blanking {pulse_onsets.size} individual stim pulses ([-{ms_before:.1f}, +{tail_ms:.1f}] ms)")
            return si.preprocessing.remove_artifacts(
                rec,
                list_triggers=pulse_onsets.tolist(),
                ms_before=ms_before,
                ms_after=tail_ms,
                mode="zeros",
            )

    return rec


def count_valid_trials_for_condition(
    sess_name: str,
    br_idx: int,
    out_base: Path,
) -> int:
    """
    Inspect existing checkpoints to count valid trials for a condition.
    Checks Aligned, PeriStim, and raw IR crossings.
    """
    trial_count = 0

    # Check PeriStim directories
    peri_dir = out_base / "checkpoints" / "PeriStim"
    if peri_dir.exists():
        cands = list(peri_dir.rglob(f"*__{sess_name}__BR_{br_idx:03d}*.npz"))
        if not cands:
            cands = list(peri_dir.rglob(f"*__BR_{br_idx:03d}*.npz"))
        for c in cands:
            if c.name.startswith("._"):
                continue
            try:
                npz = np.load(c, allow_pickle=True)
                if "event_ms" in npz and npz["event_ms"].size > 0:
                    trial_count = max(trial_count, int(npz["event_ms"].size))
                elif "trial_labels" in npz and npz["trial_labels"].size > 0:
                    trial_count = max(trial_count, int(npz["trial_labels"].size))
            except Exception:
                pass

    # Check Aligned directories
    if trial_count == 0:
        aligned_dir = out_base / "checkpoints" / "Aligned"
        if aligned_dir.exists():
            cands = list(aligned_dir.rglob(f"*__{sess_name}__*BR_{br_idx:03d}*.npz"))
            for c in cands:
                if c.name.startswith("._"):
                    continue
                try:
                    npz = np.load(c, allow_pickle=True)
                    if "ir_ms" in npz and npz["ir_ms"].size > 0:
                        trial_count = max(trial_count, int(npz["ir_ms"].size))
                    elif "stim_ms" in npz and npz["stim_ms"].size > 0:
                        trial_count = max(trial_count, int(npz["stim_ms"].size))
                except Exception:
                    pass

    # Check stim_stream.npz
    if trial_count == 0:
        stim_npz = out_base / "aux_data" / "NPRW" / f"{sess_name}_Intan_streams" / "stim_stream.npz"
        if stim_npz.exists():
            try:
                npz = np.load(stim_npz, allow_pickle=True)
                if "block_bounds_samples" in npz and npz["block_bounds_samples"].size > 0:
                    trial_count = max(trial_count, int(npz["block_bounds_samples"].shape[0]))
                elif "trigger_pairs" in npz and npz["trigger_pairs"].size > 0:
                    trial_count = max(trial_count, int(npz["trigger_pairs"].shape[0]))
            except Exception:
                pass

    return trial_count


def export_for_manual_curation(
    concat_ckpt: Path,
    min_firing_rate: float = MIN_FIRING_RATE,
    min_snr: float = MIN_SNR,
    max_rp_contamination: float = MAX_RP_CONTAMINATION,
    phy_output_folder: Optional[Path] = None,
) -> Path:
    """
    Step 1 of Manual Curation:
      1. Load analyzer_pre_slay from concat_ckpt.
      2. Filter units with Quality Metrics (FR > min_firing_rate, SNR > min_snr, RP < max_rp_contamination).
      3. Save analyzer_qm_filtered.
      4. Export to phy_for_curation (or custom phy_output_folder).
      5. Print instructions for opening Phy GUI.
    """
    _log("=" * 70)
    _log("Step 1: Preparing Dataset for Manual Curation in Phy (Quality Metric Filtering)")
    _log("=" * 70)

    analyzer_folder = concat_ckpt / "analyzer_pre_slay"
    if not analyzer_folder.exists():
        if (concat_ckpt / "analyzer_qm_filtered").exists():
            analyzer_folder = concat_ckpt / "analyzer_qm_filtered"
        elif (concat_ckpt / "sorting_curated").exists():
            analyzer_folder = concat_ckpt / "sorting_curated"
        else:
            raise FileNotFoundError(f"Cannot find analyzer_pre_slay or sorting_curated at: {concat_ckpt}")

    _log(f"Loading SortingAnalyzer from: {analyzer_folder}...")
    analyzer_precuration = si.load_sorting_analyzer(analyzer_folder)

    # Quality Metric Filtering
    _log(f"Applying QM filter (FR > {min_firing_rate} Hz, SNR > {min_snr}, RP contamination < {max_rp_contamination})...")
    qm_sess = analyzer_precuration.get_extension("quality_metrics").get_data()

    rp_col = None
    for cand in ["rp_contamination", "rp_violation", "rp_violations"]:
        if cand in qm_sess.columns:
            rp_col = cand
            break
    if rp_col is None:
        _log(f"Warning: No RP violation column found in QM dataframe ({list(qm_sess.columns)}). Skipping RP filter.")
        rp_mask = np.ones(len(qm_sess), dtype=bool)
    else:
        _log(f"Using '{rp_col}' for refractory period violation filtering.")
        rp_mask = qm_sess[rp_col] < max_rp_contamination

    fr_mask = qm_sess["firing_rate"] > min_firing_rate if "firing_rate" in qm_sess.columns else np.ones(len(qm_sess), dtype=bool)
    snr_mask = qm_sess["snr"] > min_snr if "snr" in qm_sess.columns else np.ones(len(qm_sess), dtype=bool)

    passing_mask = fr_mask & snr_mask & rp_mask
    passing = qm_sess[passing_mask].index.tolist()
    _log(f"QM Filtering complete: {len(passing)} / {len(qm_sess)} candidate units passed preliminary quality metrics.")

    analyzer_filtered = analyzer_precuration.select_units(passing)
    qm_folder = concat_ckpt / "analyzer_qm_filtered"
    if qm_folder.exists():
        shutil.rmtree(qm_folder, ignore_errors=True)
    analyzer_filtered = analyzer_filtered.save_as(folder=qm_folder, format="binary_folder")

    if phy_output_folder is None:
        phy_output_folder = concat_ckpt / "phy_for_curation"

    _log(f"Exporting QM-filtered sorting to Phy -> {phy_output_folder}...")
    export_to_phy(
        analyzer_filtered,
        output_folder=phy_output_folder,
        compute_pc_features=False,
        copy_binary=False,
        remove_if_exists=True,
    )

    _log("=" * 70)
    _log("PHY MANUAL CURATION READY!")
    _log(f"Phy folder: {phy_output_folder}")
    _log("To launch the Phy GUI for manual curation, run in your terminal:")
    _log(f'  phy template-gui "{phy_output_folder / "params.py"}"')
    _log("In Phy GUI:")
    _log("  - Review waveforms, templates, and autocorrelograms (ACG).")
    _log("  - Merge split units (select multiple units, press 'M').")
    _log("  - Split contaminated units if needed.")
    _log("  - Label units as 'good', 'mua', or 'noise' (press 'G', 'M', or 'N').")
    _log("  - Save your curation: File -> Save or Ctrl+S.")
    _log("After saving in Phy, run:")
    _log("  python preprocessing_scripts/sort_nprw_kilosort4.py --apply-curation")
    _log("=" * 70)

    return phy_output_folder


def apply_manual_curation(
    concat_ckpt: Path,
    phy_folder: Optional[Path] = None,
    spikesorting_root: Optional[Path] = None,
    exclude_noise: bool = True,
    good_only: bool = False,
) -> Path:
    """
    Step 2 of Manual Curation:
      1. Load manually curated units from Phy directory (phy_for_curation).
      2. Filter by cluster_group (exclude noise; keep good + mua, or good only).
      3. Save analyzer_manual and sorting_curated in concat_ckpt.
      4. Split continuous spike trains per session boundary into by_session/<sess_name>/NPRW/spike_trains.npz & .mat.
    """
    _log("=" * 70)
    _log("Step 2: Ingesting Manual Phy Curation into Pipeline Checkpoints...")
    _log("=" * 70)

    if phy_folder is None:
        phy_folder = concat_ckpt / "phy_for_curation"
    if not phy_folder.exists():
        if (concat_ckpt / "phy_qm_filtered").exists():
            phy_folder = concat_ckpt / "phy_qm_filtered"
        elif (concat_ckpt / "phy_slay").exists():
            phy_folder = concat_ckpt / "phy_slay"
        else:
            raise FileNotFoundError(f"Cannot find curated Phy directory at: {phy_folder}")

    _log(f"Reading manually curated sorting from: {phy_folder}...")

    # Determine cluster exclusions
    exclude_groups = ["noise"] if exclude_noise else None
    curated_sorting = se.read_phy(phy_folder, exclude_cluster_groups=exclude_groups)

    # Inspect cluster groups from cluster_group.tsv or cluster_info.tsv
    cg_file = phy_folder / "cluster_group.tsv"
    if not cg_file.exists():
        cg_file = phy_folder / "cluster_info.tsv"

    if cg_file.exists():
        try:
            df_cg = pd.read_csv(cg_file, sep="\t")
            group_col = "group" if "group" in df_cg.columns else ("KSLabel" if "KSLabel" in df_cg.columns else None)
            id_col = "cluster_id" if "cluster_id" in df_cg.columns else df_cg.columns[0]
            if group_col:
                counts = df_cg[group_col].value_counts().to_dict()
                _log(f"Phy cluster groups summary: {counts}")
                if good_only:
                    good_ids = df_cg[df_cg[group_col].astype(str).str.lower() == "good"][id_col].tolist()
                    good_ids_kept = [u for u in good_ids if u in curated_sorting.unit_ids]
                    curated_sorting = curated_sorting.select_units(good_ids_kept)
                    _log(f"Filtered strictly to 'good' clusters: {len(good_ids_kept)} units kept.")
        except Exception as e:
            _log(f"Note reading cluster group info: {e}")

    n_curated_units = curated_sorting.get_num_units()
    _log(f"Loaded {n_curated_units} manually curated units.")

    # Save sorting_curated
    curated_folder = concat_ckpt / "sorting_curated"
    if curated_folder.exists():
        shutil.rmtree(curated_folder, ignore_errors=True)
    curated_sorting.save(folder=curated_folder, format="binary_folder")
    _log(f"Saved curated SpikeInterface sorting -> {curated_folder}")

    # Build and save analyzer_manual if recording is available
    rec_corrected_saved = concat_ckpt / "rec_corrected_saved"
    rec_obj = None
    if rec_corrected_saved.exists():
        try:
            rec_obj = si.load(rec_corrected_saved)
        except Exception:
            rec_obj = None

    if rec_obj is None and (concat_ckpt / "analyzer_qm_filtered").exists():
        try:
            prev_analyzer = si.load_sorting_analyzer(concat_ckpt / "analyzer_qm_filtered")
            rec_obj = prev_analyzer.recording
        except Exception:
            rec_obj = None

    if rec_obj is None and (concat_ckpt / "analyzer_pre_slay").exists():
        try:
            prev_analyzer = si.load_sorting_analyzer(concat_ckpt / "analyzer_pre_slay")
            rec_obj = prev_analyzer.recording
        except Exception:
            rec_obj = None

    manual_analyzer_folder = concat_ckpt / "analyzer_manual"
    if manual_analyzer_folder.exists():
        shutil.rmtree(manual_analyzer_folder, ignore_errors=True)

    if rec_obj is not None:
        try:
            _log(f"Computing SortingAnalyzer for manually curated units -> {manual_analyzer_folder}...")
            analyzer_manual = si.create_sorting_analyzer(
                curated_sorting, rec_obj,
                format="binary_folder",
                folder=manual_analyzer_folder,
                overwrite=True,
            )
            analyzer_manual.compute(
                ["random_spikes", "waveforms", "templates", "noise_levels", "quality_metrics"],
            )
            _log(f"Saved analyzer_manual with {len(analyzer_manual.unit_ids)} units.")
        except Exception as e:
            _log(f"Note: Could not compute full analyzer_manual extensions: {e}")
    else:
        _log("Note: Recording object not found; saved sorting_curated without analyzer extensions.")

    # Split continuous spike trains per session boundary
    session_index_file = concat_ckpt / "session_index.pkl"
    if session_index_file.exists():
        _log("Splitting manually curated spike trains per session...")
        with open(session_index_file, "rb") as f:
            s_idx = pickle.load(f)
        sess_names_kept = s_idx["sess_names"]
        session_boundaries = s_idx["session_boundaries"]
        fs = s_idx["fs"]

        if spikesorting_root is None:
            spikesorting_root = concat_ckpt.parent

        for i, s_name in enumerate(sess_names_kept):
            t_start = int(session_boundaries[i])
            t_end = int(session_boundaries[i + 1])

            spike_trains = {}
            for uid in curated_sorting.unit_ids:
                st = curated_sorting.get_unit_spike_train(uid, segment_index=0)
                mask = (st >= t_start) & (st < t_end)
                spike_trains[str(uid)] = st[mask] - t_start

            out_sess_dir = spikesorting_root / "by_session" / s_name / "NPRW"
            out_sess_dir.mkdir(parents=True, exist_ok=True)

            savemat(out_sess_dir / "spike_trains.mat", {f"unit_{k}": v for k, v in spike_trains.items()} | {"fs": fs})
            np.savez(out_sess_dir / "spike_trains.npz", **{f"unit_{k}": v for k, v in spike_trains.items()}, fs=fs)
            _log(f"  [{s_name}] Saved curated spike trains ({len(spike_trains)} units) -> {out_sess_dir}")
    else:
        _log(f"Warning: {session_index_file} not found; skipping session spike train splitting.")

    _log("=" * 70)
    _log("Manual Curation Ingestion Completed Successfully!")
    _log(f"Next step: Run segment_sorted_units_by_condition.py to segment into PeriStim_sorted.")
    _log("=" * 70)
    return curated_folder


def postprocess_sorting(
    concat_ckpt: Path,
    spikesorting_root: Optional[Path] = None,
    min_firing_rate: float = MIN_FIRING_RATE,
    min_snr: float = MIN_SNR,
    max_rp_contamination: float = MAX_RP_CONTAMINATION,
    mode: str = "auto",  # 'auto', 'export', 'apply'
    phy_folder: Optional[Path] = None,
    good_only: bool = False,
) -> Path:
    """
    Manual curation post-processing workflow:
      - mode='export': filter with quality metrics and export to phy_for_curation
      - mode='apply': ingest curated results from Phy into pipeline
      - mode='auto': if cluster_group.tsv exists in phy_for_curation, apply curation; else export.
    """
    target_phy = phy_folder or (concat_ckpt / "phy_for_curation")
    cg_file = target_phy / "cluster_group.tsv"

    if mode == "export":
        return export_for_manual_curation(
            concat_ckpt=concat_ckpt,
            min_firing_rate=min_firing_rate,
            min_snr=min_snr,
            max_rp_contamination=max_rp_contamination,
            phy_output_folder=target_phy,
        )
    elif mode == "apply":
        return apply_manual_curation(
            concat_ckpt=concat_ckpt,
            phy_folder=target_phy,
            spikesorting_root=spikesorting_root,
            good_only=good_only,
        )
    else:
        # Auto-detect mode
        if target_phy.exists() and cg_file.exists():
            _log(f"Detected existing manual curation in: {target_phy}")
            _log("Applying manual Phy curation to pipeline checkpoints...")
            return apply_manual_curation(
                concat_ckpt=concat_ckpt,
                phy_folder=target_phy,
                spikesorting_root=spikesorting_root,
                good_only=good_only,
            )
        else:
            _log("No manual curation detected yet.")
            _log("Exporting QM-filtered dataset for manual curation in Phy...")
            return export_for_manual_curation(
                concat_ckpt=concat_ckpt,
                min_firing_rate=min_firing_rate,
                min_snr=min_snr,
                max_rp_contamination=max_rp_contamination,
                phy_output_folder=target_phy,
            )


def run_sort_nprw(
    session_loc: Optional[Path] = None,
    overwrite: bool = True,
    process_only: Optional[List[int]] = None,
    postprocess_only: Optional[bool] = None,
    curation_mode: str = "auto",  # 'auto', 'export', 'apply'
    phy_folder: Optional[Path] = None,
    good_only: bool = False,
    min_firing_rate: float = MIN_FIRING_RATE,
    min_snr: float = MIN_SNR,
    max_rp_contamination: float = MAX_RP_CONTAMINATION,
) -> Path:
    """
    Execute end-to-end spike sorting with Kilosort4 on NPRW concatenated recordings.
    """
    if postprocess_only is None:
        postprocess_only = POSTPROCESS_ONLY
    if session_loc is None:
        if SESSION_LOC is None:
            raise ValueError("SESSION_LOC must be provided or configured in params.yaml.")
        session_loc = Path(SESSION_LOC)

    out_base = session_loc / "results"
    intan_root = session_loc / "Intan"
    nprw_aux_data = out_base / "aux_data" / "NPRW"
    nprw_ckpt_root = out_base / "checkpoints" / "NPRW"

    spikesorting_root = out_base / "checkpoints" / "spikesorting"
    blanked_dir = spikesorting_root / "NPRW_blanked_sessions"
    drift_ckpt = spikesorting_root / "NPRW_drift"
    concat_ckpt = spikesorting_root / "NPRW_concat_blanking"

    spikesorting_root.mkdir(parents=True, exist_ok=True)
    blanked_dir.mkdir(parents=True, exist_ok=True)
    drift_ckpt.mkdir(parents=True, exist_ok=True)
    nprw_aux_data.mkdir(parents=True, exist_ok=True)
    concat_ckpt.parent.mkdir(parents=True, exist_ok=True)

    if postprocess_only:
        _log(f"Running manual curation workflow (mode='{curation_mode}') on: {concat_ckpt}")
        postprocess_sorting(
            concat_ckpt=concat_ckpt,
            spikesorting_root=spikesorting_root,
            min_firing_rate=min_firing_rate,
            min_snr=min_snr,
            max_rp_contamination=max_rp_contamination,
            mode=curation_mode,
            phy_folder=phy_folder,
            good_only=good_only,
        )
        return concat_ckpt

    # 1. Clean previous sorting outputs if overwrite requested
    if overwrite and concat_ckpt.exists():
        _log(f"Clearing old sorting outputs at: {concat_ckpt}")
        shutil.rmtree(concat_ckpt, ignore_errors=True)
    concat_ckpt.mkdir(parents=True, exist_ok=True)

    # 2. Global job settings for SpikeInterface
    n_jobs = PARAMS.parallel_jobs if PARAMS and hasattr(PARAMS, "parallel_jobs") else -1
    chunk = PARAMS.chunk if PARAMS and hasattr(PARAMS, "chunk") else "4s"
    si.set_global_job_kwargs(n_jobs=n_jobs, chunk_duration=chunk, progress_bar=True)

    # 3. Probe geometry and channel mapping
    geom_path = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None) if PARAMS else None
    if geom_path is None or not Path(geom_path).exists():
        default_geom = REPO_ROOT / "config" / "probes" / "ImecPrimateStimRec128_BT_corrected_091525.mat"
        if default_geom.exists():
            geom_path = default_geom
        else:
            raise FileNotFoundError(f"Probe geometry .mat file not found at {geom_path}")

    _log(f"Loading NPRW probe geometry from: {geom_path}")
    mat_probe = loadmat(Path(geom_path))
    intan_geom = {"x": mat_probe["xcoords"].ravel(), "y": mat_probe["ycoords"].ravel()}
    if "chanMap0ind" not in mat_probe:
        raise ValueError("No 0-based chanmap (chanMap0ind) in .mat geometry file.")
    intan_probe_mapping = mat_probe["chanMap0ind"].ravel()

    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(
        positions=np.c_[intan_geom["x"], intan_geom["y"]],
        shapes="square",
        shape_params={"width": 12.0},
    )
    nprw_probe.set_device_channel_indices(intan_probe_mapping)

    # Parameters
    fs = 30000.0
    radii = (30.0, 150.0)
    artrmv_ms_before = 20.0
    artrmv_tail_ms = 20.0
    if PARAMS and hasattr(PARAMS, "probes") and "NPRW" in PARAMS.probes:
        cfg_p = PARAMS.probes["NPRW"]
        radii = (float(cfg_p.get("local_radius_inner", 30.0)), float(cfg_p.get("local_radius_outer", 150.0)))
    if PARAMS and hasattr(PARAMS, "NPRW_rate_est"):
        cfg_r = PARAMS.NPRW_rate_est
        artrmv_ms_before = float(cfg_r.get("remove_ms_before", 20.0))
        artrmv_tail_ms = float(cfg_r.get("remove_tail_ms_after", 20.0))

    intan_stream = "RHS2000 amplifier channel"
    stim_stream = "Stim channel"
    if PARAMS and hasattr(PARAMS, "probes") and "NPRW" in PARAMS.probes:
        intan_stream = PARAMS.probes["NPRW"].get("neural_data_stream", intan_stream)
        stim_stream = PARAMS.probes["NPRW"].get("stim_data_stream", stim_stream)

    # 4. Discover sessions from Intan folder & metadata
    sess_folders = rcp.list_intan_sessions(intan_root)
    if not sess_folders:
        raise FileNotFoundError(f"No Intan recording folders found in {intan_root}")
    sess_names = [s.name for s in sess_folders]

    # Load metadata mapping Intan_File <-> BR_File
    meta_csv = None
    if METADATA_CSV and Path(METADATA_CSV).exists():
        meta_csv = Path(METADATA_CSV)
    else:
        meta_cands = [f for f in (session_loc / "Metadata").glob("*metadata*.csv") if not f.name.startswith("._")]
        if meta_cands:
            meta_csv = meta_cands[0]

    intan2br = {}
    if meta_csv and meta_csv.exists():
        intan2br = rcp.get_metadata_mapping(meta_csv, "Intan_File", "BR_File")

    _, idx2sess = rcp.build_session_index_map(sess_names)
    sess2idx = {v: k for k, v in idx2sess.items()}

    # 5. Session Discovery & Dynamic Trial Validation
    # KEEP valid files (n_trials > 0); DROP empty/zero-trial conditions
    _log("Step 1: Validating trials across session recording blocks...")
    valid_sessions = []
    dropped_sessions = []

    for s_folder in sess_folders:
        s_name = s_folder.name
        i_idx = sess2idx.get(s_name, -1)
        br_idx_str = intan2br.get(i_idx) if i_idx >= 0 else None
        try:
            b_idx = int(br_idx_str) if br_idx_str is not None else -1
        except Exception:
            b_idx = -1

        # Check PROCESS_ONLY filter if provided
        if process_only is not None and len(process_only) > 0:
            if b_idx not in process_only and i_idx not in process_only:
                _log(f"  [EXCLUDE] {s_name} (BR_{b_idx:03d}): not in PROCESS_ONLY.")
                continue

        # Count trials
        n_trials = count_valid_trials_for_condition(s_name, b_idx, out_base)

        if n_trials == 0:
            # Check raw IR signal or stim stream if Aligned/PeriStim not generated yet
            stim_npz_path = nprw_aux_data / f"{s_name}_Intan_streams" / "stim_stream.npz"
            if stim_npz_path.exists():
                try:
                    loaded = np.load(stim_npz_path, allow_pickle=True)
                    if "block_bounds_samples" in loaded and loaded["block_bounds_samples"].size > 0:
                        n_trials = int(loaded["block_bounds_samples"].shape[0])
                except Exception:
                    pass

        if n_trials > 0:
            _log(f"  [KEEP] {s_name} (BR_{b_idx:03d}): {n_trials} valid trials detected.")
            valid_sessions.append((s_folder, b_idx, n_trials))
        else:
            _log(f"  [DROP] {s_name} (BR_{b_idx:03d}): 0 valid trials (empty block). Excluding from sorting.")
            dropped_sessions.append((s_folder, b_idx))

    if not valid_sessions:
        raise RuntimeError("No conditions with valid trials found for spike sorting.")

    _log(f"Session Validation Summary: {len(valid_sessions)} conditions kept, {len(dropped_sessions)} conditions dropped.")

    # 6. Preprocessing & Stim Artifact Blanking per Condition
    _log("Step 2: Preprocessing, CMR filtering, and stim artifact blanking...")
    for s_folder, b_idx, n_trials in valid_sessions:
        s_name = s_folder.name
        sess_save_path = blanked_dir / s_name

        if sess_save_path.exists() and not overwrite:
            _log(f"  [{s_name}] Blanked recording already cached at {sess_save_path}. Skipping preprocessing.")
            continue

        _log(f"  [{s_name}] Loading raw Intan neural stream '{intan_stream}'...")
        rec_raw = se.read_split_intan_files(s_folder, mode="concatenate", stream_name=intan_stream, use_names_as_ids=True)
        rec_raw = rec_raw.set_probe(nprw_probe)
        rec_raw = rcp.reorder_recording_to_geometry(rec_raw, intan_probe_mapping)
        rec_raw = spre.unsigned_to_signed(rec_raw)

        # Bandpass filter (300 - 10000 Hz, float32)
        rec_bp = spre.bandpass_filter(rec_raw, freq_min=300.0, freq_max=10000.0, dtype="float32")

        # Local median CMR
        rec_cmr = spre.common_reference(rec_bp, reference="local", operator="median", local_radius=radii)

        # Artifact blanking with stim_stream.npz
        stim_npz_path = nprw_aux_data / f"{s_name}_Intan_streams" / "stim_stream.npz"
        stim_arrays = None
        if stim_npz_path.exists():
            try:
                loaded = np.load(stim_npz_path, allow_pickle=True)
                stim_arrays = {k: loaded[k] for k in loaded.files if k != "meta"}
            except Exception as e:
                _log(f"  [{s_name}] Warning: could not load stim_stream.npz: {e}")
        else:
            try:
                stim_arrays = rcp.extract_stim_npz(
                    sess=s_folder,
                    out_dir=nprw_aux_data,
                    stim_stream_name=stim_stream,
                    chanmap_perm=intan_probe_mapping,
                )
            except Exception as e:
                _log(f"  [{s_name}] Warning: stim extraction failed or no stim stream: {e}")

        rec_blanked = blank_session(rec_cmr, stim_arrays, fs, ms_before=artrmv_ms_before, tail_ms=artrmv_tail_ms)

        # Save blanked session
        if sess_save_path.exists():
            shutil.rmtree(sess_save_path, ignore_errors=True)
        rec_blanked_saved = rec_blanked.save(folder=sess_save_path, overwrite=True)
        _log(f"  [{s_name}] Saved blanked session: duration={rec_blanked_saved.get_total_duration():.1f} s")

        del rec_raw, rec_bp, rec_cmr, rec_blanked, rec_blanked_saved, stim_arrays
        gc.collect()

    # 7. Concatenation
    _log("Step 3: Concatenating all valid blanked sessions...")
    per_session_recs = []
    sess_names_kept = []
    br_indices_kept = []

    for s_folder, b_idx, n_trials in valid_sessions:
        s_name = s_folder.name
        p = blanked_dir / s_name
        if not p.exists():
            raise FileNotFoundError(f"Blanked recording missing for {s_name} at {p}")
        per_session_recs.append(si.load(p))
        sess_names_kept.append(s_name)
        br_indices_kept.append(b_idx)

    rec_concat = si.concatenate_recordings(per_session_recs)
    session_boundaries = np.concatenate([[0], np.cumsum([r.get_num_samples() for r in per_session_recs])])

    _log(f"Concatenated {len(per_session_recs)} sessions. Total duration: {rec_concat.get_total_duration():.1f} s")

    # Save session index map
    sess_index_path = concat_ckpt / "session_index.pkl"
    with open(sess_index_path, "wb") as f:
        pickle.dump(
            {
                "session_boundaries": session_boundaries,
                "sess_names": sess_names_kept,
                "br_indices": br_indices_kept,
                "fs": fs,
            },
            f,
        )
    _log(f"Saved session index: {sess_index_path}")

    # 8. Motion Estimation & Drift Correction (DREDge)
    _log("Step 4: Estimating motion with DREDge preset (compute_motion)...")
    if drift_ckpt.exists() and overwrite:
        shutil.rmtree(drift_ckpt, ignore_errors=True)
    drift_ckpt.mkdir(parents=True, exist_ok=True)

    motion, motion_info = spre.compute_motion(
        rec_concat,
        preset="dredge",
        folder=drift_ckpt,
        output_motion_info=True,
        overwrite=True,
    )

    with open(concat_ckpt / "motion_info.pkl", "wb") as f:
        pickle.dump(motion_info, f)

    _log("Interpolating motion on concatenated recording...")
    interpolated_recording = interpolate_motion(
        recording=rec_concat,
        motion=motion_info["motion"],
        border_mode="force_extrapolate",
    )

    # Cast to int16 for Kilosort4 efficiency
    interpolated_recording_i16 = spre.astype(interpolated_recording, dtype="int16")
    rec_corrected_saved = interpolated_recording_i16.save(folder=concat_ckpt / "rec_corrected_saved", overwrite=True)
    _log(f"Motion-corrected recording cached: {concat_ckpt / 'rec_corrected_saved'}")

    # 9. Kilosort4 Spike Sorting
    _log("Step 5: Running Kilosort4 spike sorting (do_correction=False)...")
    ks4_folder = concat_ckpt / "folder_KS4"
    if ks4_folder.exists():
        shutil.rmtree(ks4_folder, ignore_errors=True)

    sorting = ss.run_sorter(
        sorter_name="kilosort4",
        recording=rec_corrected_saved,
        folder=ks4_folder,
        remove_existing_folder=True,
        do_correction=False,
    )
    _log(f"Raw Kilosort4 sorting produced {len(sorting.unit_ids)} units.")

    # 10. Automated Curation
    _log("Step 6: Automated curation (removing duplicate, excess, and redundant spikes)...")
    sorting = scur.remove_duplicated_spikes(sorting)
    sorting = scur.remove_excess_spikes(sorting, rec_corrected_saved)
    sorting = scur.remove_redundant_units(sorting, align=False, remove_strategy="max_spikes")

    curated_folder = concat_ckpt / "sorting_curated"
    if curated_folder.exists():
        shutil.rmtree(curated_folder, ignore_errors=True)
    sorting.save(folder=curated_folder, overwrite=True)
    sorting = si.load(curated_folder)
    _log(f"Curated sorting retained {len(sorting.unit_ids)} distinct units.")

    # 11. SortingAnalyzer & Quality Metrics
    _log("Step 7: Computing SortingAnalyzer & quality metrics (SNR, firing rate, rp_violation)...")
    analyzer_folder = concat_ckpt / "analyzer_pre_slay"
    if analyzer_folder.exists():
        shutil.rmtree(analyzer_folder, ignore_errors=True)

    analyzer_precuration = si.create_sorting_analyzer(
        sorting,
        rec_corrected_saved,
        format="binary_folder",
        folder=analyzer_folder,
        overwrite=True,
    )
    analyzer_precuration.compute(
        [
            "random_spikes",
            "waveforms",
            "templates",
            "noise_levels",
            "spike_amplitudes",
            "unit_locations",
            "correlograms",
            "isi_histograms",
            "quality_metrics",
        ],
        extension_params={
            "correlograms": {"bin_ms": 0.5},
            "quality_metrics": {
                "metric_names": ["snr", "firing_rate", "rp_violation"],
                "metric_params": {"rp_violation": {"refractory_period_ms": 0.5}},
            },
        },
    )

    # Export to Phy
    phy_folder = concat_ckpt / "phy_pre_slay"
    export_to_phy(analyzer_precuration, output_folder=phy_folder, compute_pc_features=False, copy_binary=False, remove_if_exists=True)
    _log(f"Phy export complete -> {phy_folder}")

    # 12. Manual Curation Preparation (QM Filtering & Phy Export)
    phy_curation_folder = postprocess_sorting(
        concat_ckpt=concat_ckpt,
        spikesorting_root=spikesorting_root,
        min_firing_rate=min_firing_rate,
        min_snr=min_snr,
        max_rp_contamination=max_rp_contamination,
        mode="export",
        phy_folder=concat_ckpt / "phy_for_curation",
    )

    # Update pipeline status if available
    try:
        from RCP_analysis.python.functions.utils import update_status_cell
        update_status_cell(PARAMS.monkey, PARAMS.session, "sort_nprw", "done")
        _log("Updated data_status_reaching.csv: sort_nprw = done")
    except Exception as e:
        _log(f"Status update note: {e}")

    _log("=" * 70)
    _log("Kilosort4 Spike Sorting Pipeline Completed Successfully!")
    _log(f"Checkpoints exported to: {concat_ckpt}")
    _log(f"Cleaned Phy dataset ready for manual curation in: {phy_curation_folder}")
    _log("=" * 70)
    return concat_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Full NPRW spike sorting with Kilosort4 & DREDge motion correction (Manual Curation Workflow).")
    parser.add_argument("--session-dir", type=str, default=None, help="Path to session directory.")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing sorting results.")
    parser.add_argument("--process-only", nargs="*", type=int, default=None, help="BR indices to process exclusively.")
    parser.add_argument("--postprocess-only", action="store_true", default=None, help="Run only post-processing on existing analyzer.")
    parser.add_argument("--export-curation", "--export-phy", dest="export_curation", action="store_true", default=False, help="Export QM-filtered sorting to Phy for manual curation.")
    parser.add_argument("--apply-curation", "--apply-phy", dest="apply_curation", action="store_true", default=False, help="Apply manual Phy curation into pipeline checkpoints.")
    parser.add_argument("--phy-dir", dest="phy_dir", type=Path, default=None, help="Custom Phy directory path.")
    parser.add_argument("--good-only", dest="good_only", action="store_true", default=False, help="Keep only units marked 'good' (exclude MUA as well as noise).")
    parser.add_argument("--min-fr", type=float, default=MIN_FIRING_RATE, help=f"Minimum firing rate threshold in Hz (default: {MIN_FIRING_RATE}).")
    parser.add_argument("--min-snr", type=float, default=MIN_SNR, help=f"Minimum SNR threshold (default: {MIN_SNR}).")
    parser.add_argument("--max-rp", type=float, default=MAX_RP_CONTAMINATION, help=f"Maximum RP contamination threshold (default: {MAX_RP_CONTAMINATION}).")
    args = parser.parse_args()

    sess_dir = Path(args.session_dir) if args.session_dir else None

    curation_mode = "auto"
    if args.export_curation:
        curation_mode = "export"
    elif args.apply_curation:
        curation_mode = "apply"

    run_postprocess = True if (args.postprocess_only or args.export_curation or args.apply_curation or POSTPROCESS_ONLY) else False

    run_sort_nprw(
        session_loc=sess_dir,
        overwrite=args.overwrite,
        process_only=args.process_only,
        postprocess_only=run_postprocess,
        curation_mode=curation_mode,
        phy_folder=args.phy_dir,
        good_only=args.good_only,
        min_firing_rate=args.min_fr,
        min_snr=args.min_snr,
        max_rp_contamination=args.max_rp,
    )


if __name__ == "__main__":
    main()

