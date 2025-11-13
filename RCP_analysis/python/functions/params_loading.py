from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import yaml

# ---------------- Params model ----------------
@dataclass
class experimentParams:
    # required-ish (via YAML `paths`)
    data_root: str
    location: Optional[str] = None
    session: Optional[str]  = None

    # file locations (must be RELATIVE, if present)
    geom_mat_rel: Optional[str] = None
    metadata_rel: Optional[str] = None
    blackrock_rel: Optional[str] = None
    video_rel: Optional[str] = None
    intan_root_rel: Optional[str] = None
    output_root: Optional[str] = None

    # processing + per-probe/session config
    highpass_hz: float = 300.0
    probes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sessions: Dict[str, Any] = field(default_factory=dict)

    # runtime / chunking
    parallel_jobs: int = 8
    threads_per_worker: int = 1
    chunk: str = "1s"

    # rate estimation
    intan_rate_est: Dict[str, Any] = field(default_factory=dict)
    UA_rate_est: Dict[str, Any] = field(default_factory=dict)

    # kinematics
    kinematics: Dict[str, Any] = field(default_factory=dict)


def load_experiment_params(yaml_path: Path, repo_root: Path) -> experimentParams:
    cfg = yaml.safe_load(yaml_path.read_text()) or {}

    def expand_placeholders(obj): # Expand placeholders such as {REPO_ROOT}
        if isinstance(obj, str):
            return obj.replace("{REPO_ROOT}", str(repo_root))
        if isinstance(obj, dict):
            return {k: expand_placeholders(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand_placeholders(v) for v in obj]
        return obj

    cfg = expand_placeholders(cfg)

    # ---- paths block (single source of truth) ----
    paths = cfg.get("paths", {}) or {}
    data_root = paths.get("data_root", str(repo_root / "data"))
    location  = paths.get("location")
    session   = paths.get("session")
    geom_mat_rel = paths.get("geom_mat_rel")

    blackrock_rel  = str(Path(location) / "Blackrock")
    intan_root_rel = str(Path(location) /"Intan")
    video_rel      = str(Path(location) /"Video")
    output_root    = str(Path(location) /"results")
    metadata_rel = str(Path(location) / "Metadata" / f"{session}_metadata.csv")

    kin_cfg = dict(cfg.get("kinematics", {}) or {})
    kin_cfg["num_camera"] = kin_cfg.get("num_camera")
    kin_cfg["keypoints"] = tuple(map(str.strip, str(kin_cfg["keypoints"]).split(",")))

    # ---- assemble dataclass ----
    params = experimentParams(
        data_root=str(data_root),
        location=location,
        session=session,
        geom_mat_rel=geom_mat_rel,
        metadata_rel=metadata_rel,
        blackrock_rel=blackrock_rel,
        video_rel=video_rel,
        intan_root_rel=intan_root_rel,
        output_root=output_root,

        highpass_hz=float(cfg.get("highpass_hz", 300.0)),
        probes=cfg.get("probes", {}) or {},
        sessions=cfg.get("sessions", {}) or {},

        parallel_jobs=int(cfg.get("parallel_jobs", 4)),
        threads_per_worker=int(cfg.get("threads_per_worker", 1)),
        chunk=str(cfg.get("chunk", "1s")),

        intan_rate_est=cfg.get("intan_rate_est", {}) or {},
        UA_rate_est=cfg.get("UA_rate_est", {}) or {},
        kinematics=kin_cfg,
    )
    return params

def _join_rel(base: Path, rel: Optional[str]) -> Optional[Path]: # Adding base_root to relative paths
    if not rel:
        return None
    params = Path(rel)
    if params.is_absolute():
        raise ValueError(f"Absolute path not allowed: {rel}")
    return (base / params).resolve()

def resolve_data_root(params) -> Path:
    return Path(params.data_root).resolve()

def resolve_output_root(params) -> Path:
    base = resolve_data_root(params)
    return _join_rel(base, params.output_root) or (base / "results")

def resolve_intan_root(params) -> Path:
    base = resolve_data_root(params)
    return _join_rel(base, params.intan_root_rel) or base

def resolve_session_intan_dir(params, session_key: str) -> Path:
    base = resolve_intan_root(params)
    rel = (params.sessions or {}).get(session_key, {}).get("intan_rel")
    return _join_rel(base, rel) or base

def resolve_probe_geom_path(params, repo_root: Path, session_key: Optional[str] = None) -> Path:
    # session override
    if session_key:
        probe = (params.sessions or {}).get(session_key, {}).get("probe")
        if probe:
            rel = ((params.probes or {}).get(probe) or {}).get("mapping_mat_rel") \
                  or ((params.probes or {}).get(probe) or {}).get("geom_mat_rel")
            if rel:
                return _join_rel(repo_root, rel)

    # global fallback
    if params.geom_mat_rel:
        return _join_rel(repo_root, params.geom_mat_rel)

    raise FileNotFoundError("Missing geometry/mapping path.")