from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import socket
import yaml
import csv

# Params model
@dataclass
class experimentParams:
    data_root: str
    location: str | None
    session: str | None
    geom_mat_rel: str | None

    # processing + per-probe/session config
    highpass_hz: float = 300.0
    lowpass_hz: float = 10000.0
    probes: dict[str, dict[str, Any]] = field(default_factory=dict)
    sessions: dict[str, Any] = field(default_factory=dict)

    # runtime / chunking
    parallel_jobs: int = 8
    threads_per_worker: int = 1
    chunk: str = "1s"

    preprocessing: dict[str, Any] = field(default_factory=dict)
    # rate estimation
    NPRW_rate_est: dict[str, Any] = field(default_factory=dict)
    UA_rate_est: dict[str, Any] = field(default_factory=dict)

    Subspace_Params: dict[str, Any] = field(default_factory=dict)
    IPCA_Params: dict[str, Any] = field(default_factory=dict)
    
    kinematics: dict[str, Any] = field(default_factory=dict)
    rsa_params: dict[str, Any] = field(default_factory=dict)


def _resolve_data_root(machines_yaml_path: Path, relative_data_root: str) -> str:
    # Prepend machine's own data mount based on hostname, needs entry in machines.yaml
    if not machines_yaml_path.exists():
        raise FileNotFoundError(
            f"machines.yaml not found at {machines_yaml_path}. "
            "Add an entry for this machine's hostname."
        )
    machines_cfg = yaml.safe_load(machines_yaml_path.read_text()) or {}
    hostname = socket.gethostname()
    machines = machines_cfg.get("machines", {}) or {}
    entry = machines.get(hostname)
    prefix = entry.get("data_root_prefix")

    prefix = prefix.rstrip("/\\")
    relative_data_root = relative_data_root.lstrip("/\\")
    return f"{prefix}/{relative_data_root}"


def _get_location_session_from_status_csv(data_root: str) -> tuple[str, str]:
    """
    Read data_root/data_status_reaching.csv and find the first row where
    'Process Session?' is 'Yes'. Return that row's Location and Session.
    """
    status_csv = Path(data_root) / "data_status_reaching.csv"

    if not status_csv.exists():
        raise FileNotFoundError(f"data_status_reaching.csv not found: {status_csv}")

    with status_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required_cols = {"Process Session?", "Location", "Session"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise KeyError(
                f"Missing required column(s) in {status_csv}: {sorted(missing)}"
            )

        for row in reader:
            process_which = str(row.get("Process Session?", "")).strip().lower()

            if process_which == "yes":
                location = str(row.get("Location", "")).strip()
                session = str(row.get("Session", "")).strip()

                if not location:
                    raise ValueError(
                        f"Found Process Session? = Yes, but Location is empty in {status_csv}"
                    )
                if not session:
                    raise ValueError(
                        f"Found Process Session? = Yes, but Session is empty in {status_csv}"
                    )

                return location, session


    print(f"[RAS] No row with 'Process Session?' = 'Yes' found in {status_csv}")
    
    return "N/A", "N/A"


def load_experiment_params(
    yaml_path: Path,
    repo_root: Path,
    machines_yaml_path: Path | None = None,
    first_run: bool = False,
) -> experimentParams:
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

    # paths block
    paths = cfg.get("paths", {}) or {}
    relative_data_root = paths.get("data_root", "")
    geom_mat_rel = paths.get("geom_mat_rel")

    # resolve machine-specific prefix and combine with the lab-relative data_root
    if machines_yaml_path is None:
        machines_yaml_path = repo_root / "config" / "machines.yaml"
    data_root = _resolve_data_root(machines_yaml_path, relative_data_root)

    if first_run:
        # For the first run, we don't have a specific session yet -> needed for run_across_sessions
        location = ""
        session = ""
    else:
        location, session = _get_location_session_from_status_csv(data_root)

    kin_cfg = dict(cfg.get("kinematics", {}) or {})
    kin_cfg["num_camera"] = kin_cfg.get("num_camera")
    kin_cfg["keypoints"] = tuple(map(str.strip, kin_cfg.get("keypoints", [])))
    
    pre_cfg = dict(cfg.get("preprocessing", {}) or {})

    # ensure process_only is a list[int]
    po = pre_cfg.get("process_only", [])
    if not isinstance(po, list):
        raise TypeError("preprocessing.process_only must be a list (e.g. [1,2])")
    pre_cfg["process_only"] = [int(x) for x in po]

    # dataclass
    params = experimentParams(
        data_root=str(data_root),
        location=location,
        session=session,
        geom_mat_rel=geom_mat_rel,

        highpass_hz=float(cfg.get("highpass_hz", 300.0)),
        lowpass_hz=float(cfg.get("lowpass_hz", 10000.0)),
        probes=cfg.get("probes", {}) or {},
        sessions=cfg.get("sessions", {}) or {},

        parallel_jobs=int(cfg.get("parallel_jobs", 4)),
        threads_per_worker=int(cfg.get("threads_per_worker", 1)),
        chunk=str(cfg.get("chunk", "1s")),

        NPRW_rate_est=cfg.get("NPRW_rate_est", {}) or {},
        UA_rate_est=cfg.get("UA_rate_est", {}) or {},
        kinematics=kin_cfg,
        preprocessing=pre_cfg,
        rsa_params=cfg.get("rsa_params", {}) or {},
    )
    return params

def resolve_probe_geom_path(params, repo_root: Path, session_key: str | None) -> Path:
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