config_loading.py
=================

``RCP_analysis/python/functions/config_loading.py``

Executed once at import time. It loads ``config/params.yaml`` through
:func:`~RCP_analysis.python.functions.params_loading.load_experiment_params`,
derives every session directory from it, and creates the output directories that
the pipeline writes into. There are no functions here — the module *is* the
configuration.

Import it for its constants::

   from RCP_analysis.python.functions.config_loading import *
   # or
   import RCP_analysis as rcp
   rcp.config_loading.PERI_ROOT

Failure behaviour: if ``params.yaml`` is missing or fails to parse, a warning is
printed and ``PARAMS``, ``SESSION_LOC``, ``OUT_BASE`` and ``MONKEY`` are set to
``None`` rather than raising — so importing the package never hard-fails on a
machine that isn't configured yet.

Constants
---------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Value
   * - ``REPO_ROOT``
     - Repository root, three parents up from this file.
   * - ``PARAMS_PATH``
     - ``REPO_ROOT/config/params.yaml``.
   * - ``PARAMS``
     - The loaded ``experimentParams``, or ``None`` on failure.
   * - ``MONKEY``
     - ``PARAMS.monkey``.
   * - ``SESSION_LOC``
     - ``PARAMS.data_root / PARAMS.location``, resolved.
   * - ``METADATA_CSV``
     - ``SESSION_LOC/Metadata/<session>_metadata.csv``.
   * - ``UA_CFG``
     - ``PARAMS.probes["UA"]``.
   * - ``CAMERA_SYNC_CH``
     - ``UA_CFG["camera_sync_ch"]``, default ``134``.
   * - ``KEYPOINTS_ORDER``
     - Tuple of DLC keypoint names from ``PARAMS.kinematics``.

Input directories
-----------------

All under ``SESSION_LOC`` and created if missing:
``BR_ROOT`` (``Blackrock``), ``VIDEO_ROOT`` (``Video``), ``VOG_ROOT`` (``VOG``),
``INTAN_ROOT`` (``Intan``), ``METADATA_ROOT`` (``Metadata``).

Output directories
------------------

``OUT_BASE`` is ``SESSION_LOC/results``. Beneath it:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Path under ``OUT_BASE``
   * - ``BEHV_AUX_DATA``
     - ``aux_data/Behavior``
   * - ``NPRW_AUX_DATA``
     - ``aux_data/NPRW``
   * - ``UA_AUX_DATA``
     - ``aux_data/UA``
   * - ``BEHV_CKPT_ROOT``
     - ``checkpoints/Behavior``
   * - ``NPRW_CKPT_ROOT``
     - ``checkpoints/NPRW``
   * - ``UA_CKPT_ROOT``
     - ``checkpoints/UA``
   * - ``ALIGNED_CKPT_ROOT``
     - ``checkpoints/Aligned``
   * - ``VOG_CKPT_ROOT``
     - ``checkpoints/VOG``
   * - ``PERI_ROOT``
     - ``checkpoints/PeriStim``

.. note::

   ``config_loading`` imports ``params_loading`` with a relative import
   specifically to avoid a circular dependency with the package's top-level
   ``__init__``. Keep it that way when editing.
