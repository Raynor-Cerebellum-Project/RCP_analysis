DLC to BR Alignment
================================

Script: ``preprocessing_scripts/align_dlc_two_cams_to_br.py``

Aligns DeepLabCut (DLC) pose estimates from two cameras to the Blackrock (BR) recording timeline.

It scans for paired Cam-0 and Cam-1 files, puts both cameras on the corrected
frame numbering produced by :doc:`ocr_frame_correction`, and attaches a BR
sample number to every DLC frame so kinematics and neural data share one clock.

.. note::
   You may need to double-check which channel the sync pulse is on
   (``camera_sync_ch`` in ``config/params.yaml``).

Steps
-----

#. Scan ``VIDEO_ROOT`` for conditions that have a complete set of {Cam-0, Cam-1} × {OCR, DLC} files.
#. Mask keypoints whose likelihood (DLC) is below 0.4, setting their ``_x`` and ``_y`` to NaN, then interpolate linearly across the gaps.
#. Reindex each camera's DLC rows from ``AVI_framenum`` onto ``CORRECTED_framenum``. Frames the camera dropped become NaN rows.
#. Map the video index to a BR file through the metadata table, locate the ``.ns5`` to get the ``camera_sync``, and read rising edges to get the BR sample for each frame.
#. Write both cameras side by side to one ``.csv``.

Inputs
------

#. Paired DLC ``.csv`` files (Cam-0 and Cam-1)

   - ``DATA_ROOT/Video/``

#. Paired OCR frame-mapping ``.csv`` files (Cam-0 and Cam-1)

   - ``DATA_ROOT/Video/OCR/``

#. BR ``.ns5`` file, for the camera sync pulse

   - ``DATA_ROOT/Blackrock/``

#. Metadata table, to map ``Video_File`` to ``BR_File``

Outputs
-------

One ``.csv`` per condition, holding both camera perspectives:

- ``DATA_ROOT/results/checkpoints/Behavior/<condition>_both_cams_aligned.csv``

Columns
^^^^^^^

The header has two rows: ``cam0`` and ``cam1`` above the per-camera columns.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``CORRECTED_framenum``
     - Corrected frame number from OCR, this is used as the row index.
   * - ``cam0/ns5_sample``, ``cam1/ns5_sample``
     - BR ``.ns5`` sample number for that frame, read off the camera sync
       channel. Identical across the two cameras.
   * - ``cam0/..._<keypoint>_x``, ``_y``
     - Keypoint position in pixels, NaN where the likelihood falls below 0.4 and
       linearly interpolated across those gaps. Keypoints tracked are listed
       under ``kinematics.keypoints`` in ``config/params.yaml``.
   * - ``cam0/..._<keypoint>_likelihood``
     - DLC likelihood for that keypoint.

.. note::
   If the ``.ns5`` sync channel yields fewer pulses than there are corrected
   frames, both cameras are truncated to the number of pulses and a warning is
   printed. If the BR file cannot be paired at all, the ``.csv`` is still
   written but without ``ns5_sample``.