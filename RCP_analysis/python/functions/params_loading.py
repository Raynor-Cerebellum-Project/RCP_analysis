from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

# ---------------- Params model ----------------
@dataclass
class experimentParams:
    """
    class for yaml data
    """
    # ---- required / core ----
    data_root: str
    # session-centric inputs (may come from paths.*)
    location: Optional[str] = None  # e.g. "Nike/NRR_RW003_check"
    session: Optional[str]  = None  # e.g. "NRR_RW003_check"

    # derived or legacy (can be absent in YAML; we compute them)
    blackrock_rel: Optional[str] = None
    video_rel: Optional[str] = None
    intan_root_rel: Optional[str] = None
    metadata_rel: Optional[str] = None
    output_root: Optional[str] = None
    geom_mat_rel: Optional[str] = None

    # processing params (keep your defaults/back-compat)
    highpass_hz: float = 300.0
    probes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    probe_arrays: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    parallel_jobs: int = 4
    threads_per_worker: int = 1
    chunk: str = "1s"
    mapping_mat_rel: Optional[str] = None
    dig_line: Optional[str] = None
    stim_nums: Dict[str, int] = field(default_factory=dict)

    # thresholding config (as you had)
    intan_rate_est: Dict[str, Any] = field(default_factory=dict)
    UA_rate_est: Dict[str, Any] = field(default_factory=dict)

    # optional absolute
    intan_root: Optional[str] = None

    # sessions (referenced by your helper functions)
    sessions: Dict[str, Any] = field(default_factory=dict)

# ---------------- Loader ----------------
def load_experiment_params(yaml_path: Path, repo_root: Path) -> experimentParams:
    cfg = yaml.safe_load(yaml_path.read_text()) or {}

    # Expand {REPO_ROOT} placeholders
    def expand_placeholders(obj):
        if isinstance(obj, str):
            return obj.replace("{REPO_ROOT}", str(repo_root))
        if isinstance(obj, dict):
            return {k: expand_placeholders(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand_placeholders(v) for v in obj]
        return obj

    cfg = expand_placeholders(cfg)

    # Prefer nested "paths" block; fall back to flat keys for back-compat
    paths = cfg.get("paths", {})
    data_root = paths.get("data_root", cfg.get("data_root", str(repo_root / "data")))
    location  = paths.get("location",  cfg.get("location"))
    session   = paths.get("session",   cfg.get("session"))
    geom_mat_rel = paths.get("geom_mat_rel", cfg.get("geom_mat_rel"))

    # Legacy explicit relatives (if present, they win)
    blackrock_rel   = cfg.get("blackrock_rel")
    video_rel   = cfg.get("video_rel")
    intan_root_rel  = cfg.get("intan_root_rel")
    output_root     = cfg.get("output_root")
    metadata_rel    = cfg.get("metadata_rel")

    # If not provided explicitly, derive from location/session
    if location:
        # Derive defaults only if missing (so legacy keys still work)
        blackrock_rel  = blackrock_rel  or str(Path(location) / "Blackrock")
        video_rel  = video_rel  or str(Path(location) / "Video")
        intan_root_rel = intan_root_rel or str(Path(location) / "Intan")
        output_root    = output_root    or str(Path(location) / "results")
        # Prefer {session}_metadata.csv if session available
        default_meta = Path(location) / "Metadata" / (f"{session}_metadata.csv" if session else "metadata.csv")
        metadata_rel  = metadata_rel or str(default_meta)

    p = experimentParams(
        data_root=str(data_root),
        location=location,
        session=session,
        geom_mat_rel=geom_mat_rel,

        # derived / legacy
        blackrock_rel=blackrock_rel,
        video_rel=video_rel,
        intan_root_rel=intan_root_rel,
        metadata_rel=metadata_rel,
        output_root=output_root,

        # processing / misc (keep your previous semantics)
        highpass_hz=float(cfg.get("highpass_hz", 300.0)),
        probes=cfg.get("probes", {}) or {},
        probe_arrays=cfg.get("probe_arrays", {}) or {},
        parallel_jobs=int(cfg.get("parallel_jobs", 4)),
        threads_per_worker=int(cfg.get("threads_per_worker", 1)),
        chunk=str(cfg.get("chunk", "1s")),
        mapping_mat_rel=cfg.get("mapping_mat_rel"),
        dig_line=cfg.get("dig_line") or None,
        stim_nums=cfg.get("stim_nums", {}) or {},
        intan_rate_est=cfg.get("intan_rate_est", {}) or {},
        UA_rate_est=cfg.get("UA_rate_est", {}) or {},
        intan_root=cfg.get("intan_root"),
        sessions=cfg.get("sessions", {}) or {},
    )
    return p

# ---------------- Resolvers (unchanged) ----------------
def _resolve_path(base: Path, rel_or_abs: Optional[str]) -> Optional[Path]:
    if not rel_or_abs:
        return None
    s = str(rel_or_abs)
    return Path(s).resolve() if s.startswith("/") else (base / s).resolve()

def resolve_data_root(p) -> Path:
    return Path(p.data_root).resolve()

def resolve_output_root(p) -> Path:
    """
    Priority:
    1. Absolute p.output_root
    2. Relative to data_root
    3. Fallback: data_root / "results"
    """
    base = resolve_data_root(p)
    return _resolve_path(base, p.output_root) or (base / "results")

def resolve_intan_root(p) -> Path:
    """
    Priority:
    1. p.intan_root (absolute or relative)
    2. p.intan_root_rel (relative to data_root)
    3. Fallback: data_root
    """
    if p.intan_root:
        return _resolve_path(Path("."), p.intan_root)
    base = resolve_data_root(p)
    return _resolve_path(base, p.intan_root_rel) or base

def resolve_session_intan_dir(p, session_key: str) -> Path:
    base = resolve_intan_root(p)
    sess_cfg = (p.sessions or {}).get(session_key, {})
    return _resolve_path(base, sess_cfg.get("intan_rel")) or base

def resolve_probe_geom_path(p, repo_root: Path, session_key: Optional[str]=None) -> Path:
    if session_key:
        sess_cfg = (p.sessions or {}).get(session_key, {})
        probe_name = sess_cfg.get("probe") if sess_cfg else None
        if probe_name:
            probe_dict = (p.probes or {}).get(probe_name, {}) or {}
            rel = probe_dict.get("mapping_mat_rel") or probe_dict.get("geom_mat_rel")
            if rel:
                out = _resolve_path(repo_root, rel)
                if out:
                    return out

    rel = p.geom_mat_rel or p.mapping_mat_rel
    if rel:
        out = _resolve_path(repo_root, rel)
        if out:
            return out

    raise FileNotFoundError("No geometry/mapping file specified (probe override and global fallback both missing).")
