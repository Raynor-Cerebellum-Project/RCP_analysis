br_preproc.py
=============

``RCP_analysis/python/functions/br_preproc.py``

Blackrock-side preprocessing: enumerate sessions, pull the NS5/NS2 auxiliary and
sync streams, and apply the Utah array (UA) channel mapping including
per-electrode cortical region assignment.

.. py:currentmodule:: RCP_analysis.python.functions.br_preproc

Sessions and paths
------------------

.. py:function:: list_br_sessions(br_root)

   List Blackrock sessions under ``br_root``. A session is either a
   subdirectory, or the base filename of an NSx/NEV set sitting loose in the
   root (in which case the returned paths are extension-less bases).

.. py:function:: ua_excel_path(repo_root, probes_cfg)

   Resolve ``probes["UA"]["mapping_mat_rel"]`` against ``repo_root`` (absolute
   paths are used as-is). Returns ``None`` if the key is unset or the file does
   not exist.

Aux and sync streams
--------------------

.. py:function:: extract_br_aux_streams_npz(sess, aux_dir, camera_sync_ch, triangle_sync_ch, touchscreen_ch, hr_ch, vog_sig_ch)

   Pull the NS5 sync channels and the full NS2 stream for one session and save
   both into a single ``<aux_dir>/<session>__BR_aux_data.npz``.

   From **ns5**, the requested ``camera_sync_ch`` and ``triangle_sync_ch`` are
   stacked as rows of ``ns5_aux_traces``; channels that are absent are skipped
   with a warning. Metadata: ``ns5_session``, ``ns5_stream_name``, ``ns5_fs_hz``,
   ``ns5_channel_ids``, and the per-signal rates ``fs_camera_sync``,
   ``fs_triangle``, ``fs_touchscreen``.

   From **ns2**, every channel is stored as ``ns2_aux_traces``
   ``(n_channels, n_samples)`` and four signals are extracted **by channel
   name**, not by index:

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - ``channel_name``
        - Meaning
      * - ``hhpos``
        - VOG signal (``vog_sig``)
      * - ``vog_sync``
        - VOG sync (stored as ``ns2_vog_sync_trace``)
      * - ``touchscreen_syn``
        - Touchscreen signal
      * - ``heart_rate``
        - Heart-rate signal

   Missing streams are tolerated: an absent ``.ns5`` or ``.ns2`` prints a
   warning and contributes nothing. Nothing is written if neither is found.

   :returns: ``(touchscreen_sig, hr_sig, vog_sig, meta_ns5, meta_ns2)``. The
      caller is expected to merge ``meta_ns5``/``meta_ns2`` into a larger meta
      dict; each is ``{}`` when its stream was missing.

   .. note::

      The ``touchscreen_ch``, ``hr_ch`` and ``vog_sig_ch`` arguments are accepted
      for signature compatibility — the ns2 signals are found by name, so those
      channel numbers are not used.

Utah array mapping
------------------

.. py:function:: load_UA_mapping_from_excel(xls_path, sheet=0, n_elec=None)

   Read the UA mapping spreadsheet and return ``mapped_nsp``, an array indexed
   by ``electrode - 1`` holding the NSP channel number. Requires ``NSP ch`` and
   ``Elec#`` columns; ``ch-`` prefixes in ``NSP ch`` are stripped.

.. py:function:: apply_ua_mapping_with_regions(recording, mapped_nsp, br_idx, meta_csv, monkey, port=None)

   Rename a recording's channels to ``UAe###_NSP###`` and build the per-row UA
   arrays, including cortical region assignment.

   The UA port comes from the ``port`` argument if given, otherwise from the
   ``UA_port`` column of the ``BR_File == br_idx`` row of ``meta_csv``,
   otherwise it falls back to ``"A"`` with a warning. Port B shifts local NSP
   IDs 1–128 to 129–256 before the lookup.

   :returns: ``(renamed, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port)``

      * ``renamed`` — the recording with renamed channel IDs and an
        ``ua_row_index`` annotation.
      * ``idx_rows`` ``(n_electrodes,)`` — electrode → recording row, ``-1`` if
        unmapped.
      * ``ua_elec``, ``ua_nsp`` ``(n_channels,)`` — row → electrode number / NSP
        ID, ``-1`` if unmapped.
      * ``ua_region`` ``(n_channels,)`` — row → region index, ``-1`` if unmapped.
      * ``ua_region_names`` — ``["SMA", "Dorsal premotor", "M1 inferior", "M1 superior"]``.
      * ``ua_port`` — the port actually used.

   Region indices are the same for both animals (0 = SMA, 1 = PMd, 2 = M1i,
   3 = M1s); only the electrode ranges differ:

   .. list-table::
      :header-rows: 1
      :widths: 25 25 25

      * - Electrodes
        - Nike
        - Ada
      * - 1–64
        - SMA
        - M1 superior
      * - 65–128
        - Dorsal premotor
        - M1 inferior
      * - 129–192
        - M1 inferior
        - SMA
      * - 193–256
        - M1 superior
        - Dorsal premotor

.. py:function:: load_electrode_mapping(csv_path)

   Load the electrode mapping CSV (columns ``ElectrodeID``, ``NSP_ID``,
   ``Array``, ``GridRow``, ``GridCol``).

   :returns: ``(nsp_to_elec, region_grids, elec_to_region)`` where
      ``region_grids`` maps a region name to an 8×8 array of electrode IDs.

   Returns empty containers (with a printed warning) if the CSV is missing or
   unreadable, rather than raising.

.. py:function:: build_elec_to_data_idx(ua_ids_1based, nsp_to_elec, ua_port='A')

   Build ``{electrode_id: channel_index}``. If more than half of
   ``ua_ids_1based`` are recognizable electrode IDs for this port, they are used
   directly; otherwise it falls back to sequential channel → NSP mapping
   (channel 0 → NSP 1 for port A, NSP 129 for port B) via ``nsp_to_elec``.

.. py:function:: get_region_from_group_name(group_name)

   ``'M1s (n=45)'`` → ``'M1s'``.

.. py:function:: get_region_grid(region_name, utah_elec_grids)

   Look up a region's 8×8 grid, resolving the aliases ``M1 Inf`` → ``M1i`` and
   ``M1 Sup`` → ``M1s``. Returns ``None`` if the region is absent.
