def _load_intan_idx(sess_folder: Path):
    """
    Load Data.Intan_idx from the first '*Cal.mat' in the session folder.
    Returns a 0-based int array or None if not found.
    """
    cals = sorted(sess_folder.glob("*Cal.mat"))
    if not cals:
        return None

    M = loadmat(cals[0], squeeze_me=True, struct_as_record=False)
    Data = M.get("Data", None)
    if Data is None:
        return None

    # Robust access across MATLAB struct flavors
    idx = None
    try:
        # object-like (fields accessible as attributes)
        idx = getattr(Data, "Intan_idx", None)
    except Exception:
        pass
    if idx is None:
        # record-array / void dtype with named fields
        try:
            if isinstance(Data, np.void) and "Intan_idx" in Data.dtype.names:
                idx = Data["Intan_idx"]
        except Exception:
            pass
    if idx is None:
        return None

    idx = np.asarray(idx).ravel().astype(np.int64)
    # MATLAB -> Python indexing
    return idx - 1