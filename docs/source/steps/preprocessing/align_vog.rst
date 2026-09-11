VOG to BR Alignment
===========================

Script: ``preprocessing_scripts/align_VOG_to_br.py``

Aligns VOG to the Blackrock recording. Attaches BR ns2 sample times to VOG CSVs.

.. note::
   You may need to double-check which channel the sync pulse is on (``VOG_sync_ch`` in ``config/params.yaml``).

Steps
-----

#. Scan ``VOG_ROOT`` for ``.csv`` files. VOG index is taken from the end of the filename (e.g. ``002`` for ``NRR_RW011_002``).
#. Map that index to a BR file through the metadata table and locate the ``.ns2``. Note this step uses ``.ns2``, not the ``.ns5`` for the camera alignment.
#. Read rising edges on the VOG sync channel to get the BR sample for each VOG frame.
#. Insert ``ns2_sample`` as the first column and write the aligned ``.csv``.

Inputs
------

#. VOG ``.csv`` files

   - ``DATA_ROOT/VOG/``

#. BR ``.ns2`` file, for the VOG sync pulse

   - ``DATA_ROOT/Blackrock/``

#. Metadata table, to map ``VOG_File`` to ``BR_File``

Outputs
-------

One aligned ``.csv`` per VOG file, with BR ns2 sample times attached:

- ``DATA_ROOT/results/checkpoints/VOG/<condition>_VOG_aligned.csv``

Columns
^^^^^^^

``ns2_sample`` is added by this step; the rest are carried through from the
source VOG ``.csv`` unchanged.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Meaning
   * - ``ns2_sample``
     - BR ``.ns2`` sample number for that VOG frame, read off the VOG sync
       channel. This is converted to milliseconds assuming a 1 kHz ns2 rate.
   * - ``TimeStamp_ns``
     - VOG's own clock in nanoseconds. The step between rows gives the capture
       rate, which is not fixed across sessions (about 100 Hz in recent
       sessions, about 190 Hz in earlier ones).
   * - ``FrameNumber``
     - VOG's frame counter. It increments by exactly 1 per row but does not start at 0, so it is a VOG's own identifier.
   * - ``PCDateTime``
     - Clock time from the acquisition PC.
   * - ``HPos_pix``, ``VPos_pix``
     - Horizontal and vertical eye position in camera pixels.
   * - ``H_volt``, ``V_volt``
     - The same eye position as the analog voltage the VOG system outputs.
   * - ``H_head``, ``V_head``
     - Head position channels. These are currently just zero.

Only ``HPos_pix``, ``VPos_pix``, ``H_volt``, and ``V_volt`` are carried forward
into the aligned files by ``make_aligned_npz_and_mat.py``.

Earlier sessions
^^^^^^^^^^^^^^^^

Sessions before NRR_RW009 use a different VOG layout: ``HPos_deg`` and
``VPos_deg`` (both zero in every file checked) and ``area`` (pupil area) in
place of ``H_volt``, ``V_volt``, ``H_head``, and ``V_head``. The script's
expected-column check is written against the newer layout, so it reports the
volt and head columns as missing on those sessions. This is harmless, and only
``HPos_pix`` and ``VPos_pix`` make it downstream from them.

The script warns if any of these columns are absent from the source file, but
still writes the output.

.. note::
   If the number of sync edges does not match the number of VOG frames, both are
   clipped to the shorter of the two and a warning is printed. If the BR file
   cannot be paired at all, the ``.csv`` is still written but without
   ``ns2_sample``.
