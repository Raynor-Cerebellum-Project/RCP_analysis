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
    cfg = yaml.safe_load(yaml_path.read_text())

    def expand_placeholders(obj): # Expand placeholders such as {REPO_ROOT}
        if isinstance(obj, str):
            return obj.replace("{REPO_ROOT}", str(repo_root))
        if isinstance(obj, dict):
            return {k: expand_placeholders(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand_placeholders(v) for v in obj]
        return obj

    cfg = expand_placeholders(cfg)

    # ---- paths block ----
    paths = cfg.get("paths", {}) or {}
    data_root = paths.get("data_root", str(repo_root / "data"))
    location  = paths.get("location")
    session   = paths.get("session")
    geom_mat_rel = paths.get("geom_mat_rel")

    kin_cfg = dict(cfg.get("kinematics", {}) or {})
    kin_cfg["num_camera"] = kin_cfg.get("num_camera")
    kin_cfg["keypoints"] = tuple(map(str.strip, (kin_cfg["keypoints"])))

    # dataclass
    params = experimentParams(
        data_root=str(data_root),
        location=location,
        session=session,
        geom_mat_rel=geom_mat_rel,

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

def resolve_probe_geom_path(params, repo_root: Path, session_key: Optional[str] = None) -> Path:
    """
    Resolve the geometry/mapping .mat path.

    Priority:
      1) Session-specific probe → mapping_mat_rel or geom_mat_rel
      2) Global params.geom_mat_rel
    """
    rel = None

    # 1) Session-specific override
    if session_key:
        sessions = getattr(params, "sessions", {}) or {}
        probes   = getattr(params, "probes", {}) or {}

        sess_cfg  = sessions.get(session_key, {})
        probe_key = sess_cfg.get("probe")
        if probe_key:
            probe_cfg = probes.get(probe_key, {})
            rel = probe_cfg.get("mapping_mat_rel") or probe_cfg.get("geom_mat_rel")

    # 2) Global fallback
    if not rel:
        rel = getattr(params, "geom_mat_rel", None)

    if not rel:
        raise FileNotFoundError("Missing geometry/mapping path (no mapping_mat_rel/geom_mat_rel found).")

    # rel is always something like "config/ImecPrimateStimRec128_...mat"
    return (repo_root / rel).resolve()