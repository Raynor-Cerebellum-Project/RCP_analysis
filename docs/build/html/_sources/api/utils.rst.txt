utils.py
========

``RCP_analysis/python/functions/utils.py``

The general-purpose toolbox: status-CSV bookkeeping, OCR/DLC discovery and
loading, video-frame-to-sample sync, peri-stim segment extraction, stim-detection
NPZ readers, kinematics filtering, and spike binning / rate smoothing. Nearly
everything here is re-exported at the package top level.

.. py:currentmodule:: RCP_analysis.python.functions.utils

Status CSV and naming
---------------------

.. py:function:: short_npz_name(path)

   Shorten a peri-stim NPZ filename for logging by keeping the segment from
   ``BR_`` onward, e.g.
   ``peristim__NRR_RW022_260311_135426__BR_001_target_A.npz`` →
   ``BR_001_target_A.npz``. Falls back to the full filename if there is no
   ``BR_`` segment.

.. py:function:: update_status_cell(csv_path, row_id, column, value, id_column)

   Set one cell in ``data_status*.csv`` and rewrite the file in place.

   :param Path csv_path: the status CSV.
   :param str row_id: value in ``id_column`` identifying the row, e.g. ``"NRR_RW011"``.
   :param str column: header of the cell to write, e.g. ``"Peristim traces extracted"``.
   :param str value: new cell contents, e.g. ``"Completed"``.
   :param str id_column: header of the ID column.

OCR / DLC discovery
-------------------

.. py:function:: find_per_cond_inputs(video_root)

   Scan ``video_root/DLC`` and ``video_root/OCR`` for ``NRR_*_Cam-*.csv`` files
   and return ``{cond_id: {"ocr": {cam: Path}, "dlc": {cam: Path}}}``. When a
   condition/camera has several files, the newest by mtime wins. Only conditions
   that have **both** cameras (0 and 1) for **both** OCR and DLC are returned.

.. py:function:: _parse_condition_cam(path)

   Split a filename like ``NRR_RW007_001_[date]_Cam-1DLC_Resnet50_....csv`` into
   ``(cond_id, cam_number)``. Returns ``(None, None)`` for files with no
   ``_Cam-`` segment.

.. py:function:: get_metadata_mapping(meta_csv, field1, field2)

   Build a ``{int(field1): str(field2)}`` mapping from the session metadata CSV,
   e.g. ``BR_File`` → ``Intan_File``. Rows with a blank key or value, or a
   non-integer key, are skipped.

.. py:function:: load_ocr_map(ocr_csv)

   Load the OCR frame-correction CSV, keeping ``AVI_framenum``,
   ``OCR_framenum`` and ``CORRECTED_framenum`` as numeric columns. Warns (but
   continues) if the final ``OCR_framenum`` and ``CORRECTED_framenum`` disagree.
   Raises ``ValueError`` if there are no valid rows.

.. py:function:: load_dlc(dlc_csv)

   Load a 3-header-row DeepLabCut CSV. If a ``frame`` column is present at any
   header level it becomes the ``AVI_framenum`` index, otherwise a ``0..N-1``
   range index is used. MultiIndex columns are flattened to
   ``bodypart_coord``-style names and cast to numeric where possible.

.. py:function:: align_dlc_to_corrected(dlc_df, ocr_df)

   Reindex DLC rows from ``AVI_framenum`` into ``CORRECTED_framenum`` space
   using the OCR mapping. Frames dropped by the camera become rows of NaN.

Sync and frame-to-sample mapping
--------------------------------

.. py:function:: detect_IR_crossings(x, fs, refractory_sec=0.0005)

   Detect falling threshold crossings of an IR beam-break trace. The threshold
   is the midpoint of the signal's min and max; NaNs are filled with the median
   first so indexing is unchanged. Crossings closer together than
   ``refractory_sec`` are dropped.

   :returns: ``int64`` array of sample indices.

.. py:function:: frame2sample_br_ns5_sync(n_corrected, ns_path, sync_chan)

   Read the camera sync channel from a Blackrock NS5 file, detect rising edges
   with hysteresis (low = midpoint, high = midpoint of the midpoint and the
   max), and return the first ``n_corrected`` edge sample indices.

   If there are fewer edges than corrected frames it warns and returns the
   shorter array rather than raising.

.. py:function:: frame2sample_br_ns2_sync(n_corrected, ns_path, sync_chan)

   Same hysteresis edge detection against the NS2 (LFP) stream, used for the VOG
   sync. Unlike the NS5 version this **raises** ``RuntimeError`` when there are
   fewer edges than corrected frames.

Aligned behavior CSVs
---------------------

.. py:function:: load_behavior_npz(csv_path, num_cam)

   Read an aligned-behavior CSV and return
   ``(ns5_sample, cam0, cam0_cols, cam1, cam1_cols)``, where ``ns5_sample`` is
   ``int64`` and the camera feature blocks are ``float32``.

   Tolerant of several layouts: two-level headers (both-camera files),
   single-level headers, columns already prefixed ``cam0_`` / ``cam1_``, and
   ``ns5_sample`` variants such as ``ns5_sample_15`` or ``cam0_ns5_sample``.
   For unprefixed single-camera files with ``num_cam=2``, the camera is inferred
   from ``Cam-0`` / ``Cam-1`` in the filename (defaulting to camera 0).

.. py:function:: _read_behavior_csv_robust(csv_path, num_cam)

   Header-normalizing reader used by :func:`load_behavior_npz`; returns
   ``(df, ns5_col)`` with the ns5 column renamed to ``ns5_sample``.

.. py:function:: _flatten_cols_mi(mi)

   Flatten MultiIndex columns, dropping ``nan`` / ``Unnamed*`` levels and
   normalizing any ns5-sample-like name to ``ns5_sample``.

.. py:function:: _is_ns5_col(name)

   True for ``ns5_sample``, ``ns5sample`` and suffixed variants like
   ``ns5_sample_15``.

Intan helpers
-------------

.. py:function:: list_intan_sessions(root)

   Sorted list of session subdirectories under ``root``.

.. py:function:: save_recording(rec, out_dir)

   ``rec.save(folder=out_dir, overwrite=True)``, creating ``out_dir`` first.

.. py:function:: load_intan_aux(npz_path)

   Read an Intan aux NPZ (memory-mapped) and return
   ``(triangle_sync_signal, br_template_signal, fs_hz)`` — rows 0 and 1 of
   ``aux_traces``. If only one row is present, a zero row is appended.

Peri-stim segments
------------------

.. py:function:: extract_peristim_segments(rate_hz, counts, t_ms, stim_ms, win_ms=(-800.0, 1200.0), min_trials=1)

   Cut ``(n_ch, T)`` rate (and optional count) traces into per-trial windows
   around each stim time.

   :param rate_hz: ``(n_ch, T)`` firing rates.
   :param counts: ``(n_ch, T)`` spike counts, or None.
   :param t_ms: ``(T,)`` timebase in ms, assumed uniformly binned.
   :param stim_ms: trigger times in ms, in the same timebase.
   :param win_ms: window relative to each trigger.
   :param min_trials: raise if fewer valid trials survive.
   :returns: ``(rate_segments, count_segments, rel_time_ms, stim_ms_valid)`` with
      segments shaped ``(n_trials, n_ch, n_twin)``.

   Triggers whose window falls outside ``t_ms``, or whose slice length does not
   match the expected window length, are silently skipped — which is why
   ``stim_ms_valid`` is returned alongside.

.. py:function:: baseline_zero_each_trial(segments, rel_time_ms, normalize_first_ms=200.0)

   Subtract, per trial and channel, the mean over the first
   ``normalize_first_ms`` of the window. Raises ``ValueError`` if that window
   contains no bins.

.. py:function:: median_across_trials(zeroed_segments)

   NaN-ignoring median over the trial axis: ``(n_trials, n_ch, n_t)`` →
   ``(n_ch, n_t)``. All-NaN bins stay NaN.

.. py:function:: variance_across_trials(zeroed_segments)

   NaN-ignoring variance over the trial axis, same shapes as above.

Stim detection files
--------------------

.. py:function:: load_stim_detection(npz_path)

   Read a ``stim_stream.npz`` and return a dict with ``trigger_pairs``
   ``(n_pulses, 2)``, ``block_bounds_samples`` ``(n_blocks, 2)``,
   ``pulse_sizes`` ``(n_pulses,)`` and ``active_channels`` (empty if absent).
   Sample indices are in Intan index space.

.. py:function:: stim_npz_path_from_br_idx(br_idx, mapping_csv, nprw_aux_root)

   Map a Blackrock file index to its Intan session and return
   ``(stim_npz_path, session_name)``.

   ``BR_File`` → ``Intan_File`` comes from the metadata CSV; the session name is
   the *n*-th ``*_Intan_streams`` directory under ``nprw_aux_root``
   (``Intan_File`` is 1-based). Returns ``(None, None)`` — with a printed warning
   — if the CSV, the mapping, or the session index cannot be resolved.

.. py:function:: detect_stim_channels_from_npz(stim_npz_path, eps=1e-12, min_edges=1)

   Count 0→nonzero rising edges per channel across the whole ``stim_traces``
   array and return the geometry-ordered indices of channels with at least
   ``min_edges``. Per-channel threshold is the midpoint of the 5th and 95th
   percentiles; channels with no finite data are ignored. Returns an empty array
   if the file or the ``stim_traces`` key is missing.

.. py:function:: aligned_stim_ms(stim_ms_abs, meta)

   Shift absolute Intan stim times into the aligned timebase of a combined file,
   using ``meta["shift_ms"]`` or, failing that,
   ``meta["shift_sample"] * 1000 / meta["fs_intan"]``. Returns the input
   unchanged when neither key is present.

Session and file lookup
-----------------------

.. py:function:: parse_intan_session_dtkey(session)

   Extract the trailing ``YYMMDD_HHMMSS`` from a session name as a sortable int;
   returns ``999999999999`` when the pattern is absent, so unparsable sessions
   sort last.

.. py:function:: build_session_index_map(intan_sessions)

   Sort sessions chronologically and return ``({session: index}, {index: session})``
   with 1-based indices matching the ``Intan_File`` column.

.. py:function:: find_ns5_by_br_index(br_root, br_idx)
.. py:function:: find_ns2_by_br_index(br_root, br_idx)

   Recursively find the ``.ns5`` / ``.ns2`` file whose stem ends with the
   zero-padded 3-digit BR index (``br_idx=1`` → ``*001.ns5``). On multiple hits,
   prefer the newest mtime, then the largest file. Returns ``None`` with a
   warning if there is no match.

.. py:function:: ua_title_from_meta(meta)

   Plot title for Utah array figures: ``"Blackrock / UA: NRR_RW_001_<br_idx>"``,
   or ``"Utah/BR"`` when ``br_idx`` is absent.

Kinematics filtering
--------------------

.. py:function:: butter_lowpass_pos_and_vel(TxD, t_ms, cutoff_hz=10.0, order=3)

   Zero-phase Butterworth low-pass of a ``(T, D)`` position array plus a central
   difference velocity. Returns ``(pos_filt, vel, t_ms)``. The sample rate is
   derived from the median ``diff(t_ms)``; columns too short for ``filtfilt``
   are passed through unfiltered. Velocity units are position units per ms.

.. py:function:: butter_lowpass_pos_and_vel_3d(NKT, t_ms, cutoff_hz=10.0, order=3)

   Same as above for ``(n_trials, n_keypoints, T)`` arrays, filtering each
   trial × keypoint trace independently.

.. py:function:: _butter_lowpass_ba(cutoff_hz, fs_hz, order=3)

   Butterworth low-pass coefficients with the normalized cutoff clamped into
   ``(0, 1)``.

.. py:function:: _filtfilt_nanaware(x, b, a)

   ``filtfilt`` (Gustafsson's method) on a trace containing NaNs: NaNs are
   filled with the median before filtering and restored afterwards.

.. py:function:: _central_diff_ms(x, dt_ms)

   Central-difference derivative, one-sided at the edges. Any sample whose
   neighbour is NaN yields NaN.

Spike binning and rate estimation
---------------------------------

.. py:function:: dedup_peaks(peaks, amps, dedup_ms=0.5, max_cluster_ms=1.0)

   Collapse near-simultaneous detections per channel, keeping the largest
   ``|amplitude|`` in each cluster. A cluster ends when the ISI exceeds
   ``dedup_ms`` or the cluster spans more than ``max_cluster_ms``.

   :param peaks: ``{channel: spike_times_ms}``
   :param amps: ``{channel: amplitudes}``, same keys and lengths.
   :returns: ``(peaks_dedup, amps_dedup)`` in the same dict form.

.. py:function:: bin_counts_around_stim(peaks_ms, bin_ms, stim_times_ms, art_before_ms, art_after_ms, win_ms=(-600, 600), step_ms=None)

   Bin spikes around each stim event, leaving a blanking gap of
   ``[-art_before_ms, art_after_ms]`` for the artifact. Bin centres walk outward
   from the edges of that gap in both directions.

   ``step_ms=None`` (the default) gives contiguous non-overlapping bins of width
   ``bin_ms``. ``step_ms < bin_ms`` gives sliding, overlapping windows;
   ``step_ms > bin_ms`` leaves gaps between bins.

   :returns: ``(counts, bin_centers_ms, bin_widths_ms, n_left_bins)`` with
      ``counts`` shaped ``(n_trials, n_channels, n_bins)``; channels are ordered
      by ``sorted(peaks_ms)`` and ``n_left_bins`` is the index of the first bin
      after the blanking gap.

.. py:function:: smooth_counts_gauss(counts, bin_widths_ms, bin_centers_ms, sigma_ms, left_bins)

   Convert counts to rates in Hz and Gaussian-smooth them in milliseconds,
   independently on each side of the stim blanking gap so the artifact window is
   never smoothed across. NaN bins are excluded from the local normalization
   rather than treated as zeros.

   :param left_bins: number of bins before the gap, from
      :func:`bin_counts_around_stim`.
   :returns: array shaped like ``counts``, in Hz.

.. py:function:: _smooth_segment(seg_counts, seg_widths_ms, seg_centers, sigma_ms)

   One-sided worker for :func:`smooth_counts_gauss`: counts → rates, then a
   NaN-aware Gaussian weighted average over bin centres.
