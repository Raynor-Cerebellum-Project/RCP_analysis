from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import socket
import yaml
import csv
import os
import json

# Params model
@dataclass
class experimentParams:
    data_root: str
    monkey: str
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


def _get_monkey_from_data_root(data_root: str) -> str:
    """
    Identify the monkey from the final part of data_root.

    Example:
        /some/path/Nike -> Nike
        /some/path/Ada  -> Ada
    """
    monkey = Path(data_root).name.strip()

    if monkey not in {"Nike", "Ada"}:
        raise ValueError(
            f"Could not identify monkey from data_root: {data_root}. "
            f"Expected final folder to be 'Nike' or 'Ada', got '{monkey}'."
        )

    return monkey


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
    """_summary_

    Args:
        yaml_path (Path): _description_
        repo_root (Path): _description_
        machines_yaml_path (Path | None, optional): _description_. Defaults to None.
        first_run (bool, optional): _description_. Defaults to False.

    Raises:
        RuntimeError: _description_
        TypeError: _description_

    Returns:
        experimentParams: _description_
    """
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
    monkey = _get_monkey_from_data_root(data_root)

    if first_run:
        location = ""
        session = ""
    else:
        env_location = os.environ.get("RCP_LOCATION")
        env_session = os.environ.get("RCP_SESSION")

        if env_location and env_session:
            location = env_location
            session = env_session
        elif env_location or env_session:
            raise RuntimeError(
                "Both RCP_LOCATION and RCP_SESSION must be set together, "
                f"got RCP_LOCATION={env_location!r}, RCP_SESSION={env_session!r}"
            )
        else:
            location, session = _get_location_session_from_status_csv(data_root)

    kin_cfg = dict(cfg.get("kinematics", {}) or {})
    kin_cfg["num_camera"] = kin_cfg.get("num_camera")
    kin_cfg["keypoints"] = tuple(map(str.strip, kin_cfg.get("keypoints", [])))
    
    pre_cfg = dict(cfg.get("preprocessing", {}) or {})

    # ensure process_only is a list[int]
    # Priority:
    #   1) RCP_PROCESS_ONLY environment variable
    #   2) preprocessing.process_only from params.yaml
    env_process_only = os.environ.get("RCP_PROCESS_ONLY")

    if env_process_only is not None:
        try:
            po = json.loads(env_process_only)
        except json.JSONDecodeError as e:
            raise ValueError(
                "RCP_PROCESS_ONLY must be valid JSON, e.g. '[]' or '[1, 2, 5]'. "
                f"Got: {env_process_only!r}"
            ) from e
    else:
        po = pre_cfg.get("process_only", [])

    if po is None:
        po = []

    if not isinstance(po, list):
        raise TypeError(
            "process_only must be a list, either from preprocessing.process_only "
            "or RCP_PROCESS_ONLY, e.g. [] or [1, 2]."
        )

    pre_cfg["process_only"] = [int(x) for x in po]

    # dataclass
    params = experimentParams(
        data_root=str(data_root),
        monkey=monkey,
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