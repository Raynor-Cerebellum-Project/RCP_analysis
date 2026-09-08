reach_onset_detection.py
========================

``RCP_analysis/python/functions/reach_onset_detection.py``

Detects when each reach *begins*, from the behavioral kinematics. The detector
works backward from peak speed and accepts the first sample that satisfies three
criteria at once:

#. **Speed** — below a fraction of that trial's peak speed.
#. **Position** — within ``start_pos_max_thresh`` of zero (the hand is still at
   the start position).
#. **Stability** — the mean speed over the preceding window is also below
   threshold, so a momentary dip mid-reach is not mistaken for the onset.

This module is not re-exported at the package top level; import it directly::

   from RCP_analysis.python.functions.reach_onset_detection import (
       detect_reach_onset, detect_reach_onset_batch,
   )

.. py:currentmodule:: RCP_analysis.python.functions.reach_onset_detection

Configuration
-------------

.. py:class:: ReachOnsetConfig

   All tunable parameters. ``DEFAULT_CONFIG`` is a module-level instance used
   whenever ``config=None`` is passed.

   .. list-table::
      :header-rows: 1
      :widths: 34 12 54

      * - Field
        - Default
        - Meaning
      * - ``start_pos_max_thresh``
        - 0.2
        - Onset position must be within ±this of zero.
      * - ``low_speed_thresh_ratio``
        - 0.05
        - "Near zero" speed, as a fraction of the trial's peak speed.
      * - ``stable_window_ms``
        - 20.0
        - How long speed must already have been low.
      * - ``max_reach_duration_ms``
        - 800.0
        - Search window length from ``search_start_idx``.
      * - ``min_reach_duration_ms``
        - 100.0
        - Onset-to-peak time below this marks the trial invalid.
      * - ``velocity_smooth_window_ms``
        - 75.0
        - Savitzky–Golay window for the derivative.
      * - ``savgol_polyorder``
        - 3
        - Savitzky–Golay polynomial order.

Results
-------

.. py:class:: ReachOnsetResult

   Per-trial detection result.

   :Fields:
      * ``onset_time`` (float | None), ``onset_idx`` (int | None)
      * ``peak_speed_time``, ``peak_speed_idx``, ``peak_speed`` — the reference
        peak the backward scan started from.
      * ``speed_threshold`` — the absolute threshold actually used.
      * ``method`` — how the onset was found (see below).
      * ``is_valid`` (bool), ``rejection_reason`` (str).

   ``method`` values, in the order they are attempted:

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Value
        - Meaning
      * - ``speed_stable``
        - Speed, position and stability criteria all met.
      * - ``start_of_trial``
        - Criteria met, but too close to the start for a stability window.
      * - ``speed_only_fallback``
        - Position constraint dropped; speed and stability only.
      * - ``search_start_fallback``
        - Nothing found; ``search_start_idx`` used.
      * - ``failed``
        - Input arrays unusable or no valid speed data.

   ``is_valid`` is set from the minimum-duration check, so a trial can carry a
   fallback onset *and* ``is_valid=True``; check ``method`` too if you only want
   clean detections.

Core detection
--------------

.. py:function:: detect_reach_onset(position, speed, time_axis, dt, config=None, search_start_idx=0)

   Detect the onset for a single trial.

   :param position: ``(T,)`` position trace, typically the X coordinate of the
      tracked keypoint.
   :param speed: ``(T,)`` absolute velocity.
   :param time_axis: ``(T,)`` times in ms.
   :param float dt: time step in ms.
   :param ReachOnsetConfig config: defaults to ``DEFAULT_CONFIG``.
   :param int search_start_idx: index to start searching from.
   :returns: :class:`ReachOnsetResult`.

   Returns a ``failed`` result (rather than raising) when ``T < 10``, the array
   lengths disagree, or the search window holds no finite speed.

.. py:function:: detect_reach_onset_batch(pos_segs, speed_segs, time_axis, config=None, keypoint_idx=0)

   Run the detector over many trials.

   :param pos_segs: ``(n_trials, n_keypoints, T)`` or ``(n_trials, T)``.
   :param speed_segs: same shape as ``pos_segs``.
   :param time_axis: ``(T,)`` in ms; ``dt`` is inferred from it, and a ``dt``
      below 0.9 is assumed to be seconds and converted to ms.
   :param int keypoint_idx: which keypoint to use for 3-D inputs.
   :returns: ``(onset_times, onset_indices, results)`` — NaN and ``-1``
      respectively for trials whose result was not valid, plus the full list of
      :class:`ReachOnsetResult`.

Speed computation
-----------------

.. py:function:: compute_speed_from_position(position, dt, config=None)

   Speed (``|velocity|``) from a 1-D position trace, via a Savitzky–Golay
   derivative. NaNs are linearly interpolated first; traces shorter than 7
   samples, or a failed filter fit, fall back to ``np.gradient``.

.. py:function:: compute_2d_speed(x, y, dt, config=None)

   Magnitude of the 2-D velocity vector from X and Y traces. Both are smoothed
   with :func:`smooth_1d` (50 ms) and differentiated with ``np.gradient``, which
   preserves sign so the components combine correctly.

.. py:function:: smooth_1d(arr, dt, window_ms=50.0, polyorder=3)

   Savitzky–Golay smoothing of a 1-D array, NaN-interpolated first. Falls back
   to a Gaussian filter when the window would exceed the array length or the fit
   fails; arrays shorter than 5 samples are returned unchanged.

Alignment utilities
-------------------

.. py:function:: align_traces_to_onset(traces, onset_indices, time_axis, pre_onset_ms=200.0, post_onset_ms=600.0)

   Re-cut ``(n_trials, T)`` traces onto a common onset-relative time axis.

   :returns: ``(aligned_traces, aligned_time)`` where ``aligned_time`` runs from
      ``-pre_onset_ms`` to ``+post_onset_ms``. Trials with ``onset_idx < 0`` are
      left all-NaN; partial windows are NaN-padded rather than dropped.

.. py:function:: get_pre_post_regions(onset_time, time_axis, pre_duration_ms=200.0, post_duration_ms=600.0)

   Boolean masks for the pre-reach window ``[onset - pre, onset)`` and the
   during-reach window ``[onset, onset + post)``.

   :returns: ``(pre_mask, post_mask)``.
