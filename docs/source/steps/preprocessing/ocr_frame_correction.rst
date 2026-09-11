OCR Frame Correction
====================

Script: ``preprocessing_scripts/OCR_frame_correction.py``

Because FLIR cameras could drop frames, OCR is used to get a mapping of the frames to the actual times.

Each frame carries a text band burned in by the camera, holding its own frame
counter and the time since the previous captured frame. Reading that band back
recovers which frames went missing, so a position in the ``.avi`` file can be
mapped to a position on the true recording timeline.

Steps
-----

#. Get the text band and crop at the top of the frame, convert to grayscale, threshold,
   and dilate so EasyOCR reads it reliably.
#. Read the text. We expect three fields: the frame counter, a ``dt: ... msec`` interval, and trailing text.
#. Derive the dropped-frame count from ``delta_t``, taking one  frame as 10 ms (100 fps)
#. Accumulate the running total of dropped frames and add it to the absolute ``.avi`` frame index to get the corrected frame number ``CORRECTED_framenum = AVI_framenum + total_missing``.
#. Write one row per frame to the output ``.csv``.

Inputs
------

#. Two camera ``.avi`` files

   - ``DATA_ROOT/Video/``

Outputs
-------

``.csv`` files containing the mapping of frames to correct frames/times, one per
video, mirroring the input subdirectory structure:

- ``DATA_ROOT/Video/OCR/<subfolders>/<video stem>_ocr.csv``

Columns
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``AVI_framenum``
     - Absolute frame number in the video.
   * - ``OCR_framenum``
     - Frame number read by OCR, which skips over dropped frames.
   * - ``CORRECTED_framenum``
     - Corrected frame number (``AVI_framenum + total_missing``).
   * - ``delta_t``
     - Difference in time between previous frame and current frame in ms.
   * - ``n_missing``
     - Frames dropped immediately before current frame.
   * - ``total_missing``
     - Cumulative frames dropped.
   * - ``n_chunks_missing``
     - Number of separate "drop events" so far.
   * - ``all_text``
     - Raw OCR text, kept for debugging.
