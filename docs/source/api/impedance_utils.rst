impedance_utils.py
==================

``RCP_analysis/python/functions/impedance_utils.py``

Parses the impedance measurement files saved at the start of a session and
returns the channels that should be excluded. Both parsers are deliberately
forgiving: a missing file, a malformed line or an unparsable value is skipped
rather than raising, so a partially-written impedance file never blocks
preprocessing.

.. py:currentmodule:: RCP_analysis.python.functions.impedance_utils

Functions
---------

.. py:function:: get_session_impedances(session_loc, utah_high_thresh=800, utah_exclude_low=True, intan_high_thresh=4.5e6)

   Look for an ``Impedances`` folder under ``session_loc`` and parse the two
   standard files, ``Utah_imp_start_PortA`` and ``Intan_imp_start.csv``.

   :param Path session_loc: session root.
   :param float utah_high_thresh: max Utah impedance in kΩ.
   :param bool utah_exclude_low: also exclude Utah channels at or below 15 kΩ.
   :param float intan_high_thresh: max Intan impedance in Ω.
   :returns: ``{'utah': set(bad_ids), 'nprw': set(bad_ids)}``; empty sets when
      ``session_loc`` is ``None``, the folder is absent, or a file is missing.

.. py:function:: _get_bad_channels_utah(file_path, high_threshold=800, exclude_low_imp=True)

   Parse a Utah array impedance file into a set of excluded channel IDs. The
   channel ID is the part after the dash in ``elec2-124``; comment lines
   (``*``) and the ``Chan`` header are skipped.

   Excluded when the impedance is above ``high_threshold`` kΩ, or — with
   ``exclude_low_imp`` — at or below 15 kΩ, which usually means a shorted
   electrode. Both file formats are handled::

      elec2-124 15 kOhm
      elec4-245 <= 15kOhm

.. py:function:: _get_bad_channels_intan(file_path, high_threshold=4.5e6)

   Parse an Intan impedance CSV into a set of excluded channel IDs. The channel
   name is column 1 (e.g. ``NPXL-015``) and the impedance in ohms is column 4;
   the ID is the last three digits of the name. Excluded when the impedance is
   above ``high_threshold`` Ω.
