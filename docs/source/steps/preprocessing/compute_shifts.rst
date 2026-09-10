Compute BR-to-Intan Shifts
==========================

Script: ``preprocessing_scripts/compute_br_to_intan_shifts.py``

Computes the timeshift between the Intan and Blackrock recordings.

Steps
-----

#. Load metadata to match Intan files to BR files.
#. Load template sent from BR to Intan and the Intan ADC file.
#. Match the template to the BR template signal.
#. Save shifts, adjusted shifts, and related information to ``DATA_ROOT/Metadata/br_to_intan_shifts.csv``.

Inputs
------

#. Metadata table, to map ``Intan_File`` to ``BR_File``

#. Alignment template sent from BR to Intan

   - ``config/br_intan_align_template.mat``

#. Intan auxiliary streams, holding the ADC triangle signal

   - ``DATA_ROOT/results/aux_data/NPRW/<session>_Intan_streams/aux_streams.npz``

#. BR ``.ns5`` file, for the triangle sync channel

   - ``DATA_ROOT/Blackrock/``

Outputs
-------

Alignment summary
^^^^^^^^^^^^^^^^^

``DATA_ROOT/Metadata/br_to_intan_shifts.csv``
   One row per Intan/BR pair. This is the table every later step reads to put
   Intan and BR on a common clock.

   .. list-table::
      :header-rows: 1
      :widths: 35 65

      * - Column
        - Meaning
      * - ``intan_filename``, ``intan_idx``
        - Intan session name and its numeric index.
      * - ``br_filename``, ``br_idx``
        - Paired BR file name and its numeric index.
      * - ``is_control``, ``is_at_rest``, ``is_continuous_stim``
        - Condition flags carried over from the metadata table.
      * - ``notes``
        - Free-text note for the pair, from the metadata table.
      * - ``fs_intan``, ``fs_br``
        - Sampling rates of the two systems.
      * - ``shift_sample``
        - The refined shift, in Intan samples. This is the value to apply.
      * - ``shift_seconds``, ``shift_ms``
        - The same shift in seconds and milliseconds
          (``shift_sample / fs_intan``).
      * - ``dur_intan_sec``, ``dur_br_sec``
        - Recording durations, useful as a sanity check that the two files
          really are the same session.
      * - ``triangle_refined_from``
        - Where the template match initially landed, before refinement.
      * - ``triangle_refine_delta_samples``
        - How far refinement moved it, in samples
          (``shift_sample - triangle_refined_from``).
      * - ``n_locs``
        - Number of template blocks found in the Intan ADC signal.
      * - ``adc_npz``, ``locs_csv``
        - Paths to the inputs and per-block output this row was derived from.

Per-block detail
^^^^^^^^^^^^^^^^

``DATA_ROOT/Metadata/template_locs/intan_<idx>__template_locs.csv``
   One row per template block, written per Intan file.

   .. list-table::
      :header-rows: 1
      :widths: 35 65

      * - Column
        - Meaning
      * - ``block_idx``
        - Index of the template block within the recording.
      * - ``lock_loc_sample``
        - Where the template match landed for this block, before refinement.
      * - ``triangle_loc_sample``
        - The refined location, after matching against the BR triangle signal.
      * - ``delta_samples``
        - Refinement correction for this block
          (``triangle_loc_sample - lock_loc_sample``).

.. note::
   Every block is refined and written to the per-block CSV, but the summary row
   currently takes its shift from **block 0 only** (see the ``TODO Just using
   the first one now`` in the script). The remaining blocks are useful for
   checking that the shift is stable across the recording — the script prints
   the min, median, and max of ``delta_samples`` for that purpose.
