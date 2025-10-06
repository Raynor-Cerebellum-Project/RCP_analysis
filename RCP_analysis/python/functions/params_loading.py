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
    
    Attributes:
    data_root : str
        Root directory
    blackrock_rel : str
        Relative path to Blackrock session
    highpass_hz : float
        Cutoff frequency for high-pass filtering neural data
    probes : Dict[str, Dict[str, Any]]
        Mapping of probe names to metadata/config dicts (e.g. UA, Neuropixels)
    probe_arrays : Dict[str, Dict[str, Dict[str, Any]]]
        Nested dict for array-specific probe metadata
        
    Processing metadata:
    parallel_jobs : int
        Number of jobs to use in parallel (n_jobs for SpikeInterface)
    threads_per_worker : int
        Threads to allocate per worker when parallelizing
    chunk : str
        Chunk duration string (e.g. "0.5s") for processing

    Optional
    --------
    mapping_mat_rel : Optional[str]
        Relative path to .mat mapping file (alternative to Excel)
    dig_line : Optional[str]
        Digital line name/id used for trigger detection
    stim_nums : Dict[str, int]
        Mapping of stim labels -> numbers (per user config)
    """
    # ---- required ----
    data_root: str
    blackrock_rel: str
    highpass_hz: float
    probes: Dict[str, Dict[str, Any]]
    probe_arrays: Dict[str, Dict[str, Dict[str, Any]]]
    parallel_jobs: int
    threads_per_worker: int
    chunk: str
    mapping_mat_rel: Optional[str] = None
    dig_line: Optional[str] = None
    stim_nums: Dict[str, int] = field(default_factory=dict)
    intan_rate_est: Dict[str, Any] = field(default_factory=dict)
    UA_rate_est: Dict[str, Any] = field(default_factory=dict)
    # Intan-specific
    intan_root_rel: Optional[str] = None
    intan_root: Optional[str] = None
    geom_mat_rel: Optional[str] = None
    
    output_root: Optional[str] = None

# Loading params.yaml
def load_experiment_params(yaml_path: Path, repo_root: Path) -> experimentParams:
    """
    Load yaml data into the data class experimentParams.
    
    Input:
    yaml_path : Path
        Path to params.yaml
    repo_root : Path
        Root of the repo
    Returns:
        experimentParams class
    """
    cfg = yaml.safe_load(yaml_path.read_text()) or {}

#   Expand {REPO_ROOT} placeholders in yaml file so that users can have their own file paths
    def expand_placeholders(obj):
        if isinstance(obj, str):
            return obj.replace("{REPO_ROOT}", str(repo_root))
        if isinstance(obj, dict):
            return {k: expand_placeholders(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand_placeholders(v) for v in obj]
        return obj

    cfg = expand_placeholders(cfg)

    return experimentParams(
        data_root=cfg.get("data_root", str(repo_root / "data")),
        blackrock_rel=cfg["blackrock_rel"],
        highpass_hz=float(cfg["highpass_hz"]),
        probes=cfg.get("probes", {}),
        probe_arrays=cfg.get("probe_arrays", {}),
        parallel_jobs=int(cfg.get("parallel_jobs", 4)),
        threads_per_worker=int(cfg.get("threads_per_worker", 1)),
        chunk=str(cfg.get("chunk", "1s")),
        # optional
        mapping_mat_rel=cfg.get("mapping_mat_rel"),
        dig_line=cfg.get("dig_line") or None,
        stim_nums=cfg.get("stim_nums", {}) or {},
        
        # Thresholding
        intan_rate_est=cfg.get("intan_rate_est", {}) or {},
        UA_rate_est=cfg.get("UA_rate_est", {}) or {},
        # Intan-specific
        intan_root_rel=cfg.get("intan_root_rel"),
        intan_root=cfg.get("intan_root"),
        geom_mat_rel=cfg.get("geom_mat_rel"),
        output_root=cfg.get("output_root"),
    )

def _resolve_path(base: Path, rel_or_abs: Optional[str]) -> Optional[Path]:
    """
    Resolve a path, handling both absolute and relative cases.
    Returns None if rel_or_abs is falsy.
    """
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
    """
    Get Intan directory for a specific session. Falls back to global intan_root.
    """
    base = resolve_intan_root(p)
    sess_cfg = (p.sessions or {}).get(session_key, {})
    return _resolve_path(base, sess_cfg.get("intan_rel")) or base

def resolve_probe_geom_path(p, repo_root: Path, session_key: Optional[str]=None) -> Path:
    """
    Priority:
    1. Session-level probe mapping/geom override (if session_key provided)
    2. Global p.geom_mat_rel or p.mapping_mat_rel
    """
    if session_key:
        sess_cfg = (p.sessions or {}).get(session_key, {})
        probe_name = sess_cfg.get("probe") if sess_cfg else None
        if probe_name:
            probe_dict = (p.probes or {}).get(probe_name, {}) or {}
            rel = probe_dict.get("mapping_mat_rel") or probe_dict.get("geom_mat_rel")
            if rel:
                return _resolve_path(repo_root, rel)

    rel = p.geom_mat_rel or p.mapping_mat_rel
    if rel:
        return _resolve_path(repo_root, rel)

    raise FileNotFoundError("No geometry/mapping file specified (probe override and global fallback both missing).")
