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
        
    Plotting params:
    win_pre_s : float
        Window length before stimulation (seconds)
    win_post_s : float
        Window length after stimulation (seconds)
    stim_bar_ms : float
        Duration of stimulation for plotting Ex: 100ms
    quicklook_rows : int
        Rows in quicklook plotting grid - unused for now
    quicklook_cols : int
        Columns in quicklook plotting grid - unused for now
    stride : int
        Stride (samples or frames) for downsampling quicklook plots
    default_stim_num : int
        Default stimulation channel/number to use if unspecified
    sessions : Dict[str, Dict[str, Any]]
        Dictionary of per-session overrides or metadata

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
    win_pre_s: float
    win_post_s: float
    stim_bar_ms: float
    quicklook_rows: int
    quicklook_cols: int
    stride: int
    default_stim_num: int
    sessions: Dict[str, Dict[str, Any]]

    # ---- optional ----
    mapping_mat_rel: Optional[str] = None
    dig_line: Optional[str] = None
    stim_nums: Dict[str, int] = field(default_factory=dict)


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
        parallel_jobs=int(cfg.get("parallel_jobs", 1)),
        threads_per_worker=int(cfg.get("threads_per_worker", 1)),
        chunk=str(cfg.get("chunk", "0.5s")),
        win_pre_s=float(cfg["win_pre_s"]),
        win_post_s=float(cfg["win_post_s"]),
        stim_bar_ms=float(cfg["stim_bar_ms"]),
        quicklook_rows=int(cfg["quicklook_rows"]),
        quicklook_cols=int(cfg["quicklook_cols"]),
        stride=int(cfg["stride"]),
        default_stim_num=int(cfg["default_stim_num"]),
        sessions=cfg.get("sessions", {}) or {},
        # optional
        mapping_mat_rel=cfg.get("mapping_mat_rel"),
        dig_line=cfg.get("dig_line") or None,
        stim_nums=cfg.get("stim_nums", {}) or {},
    )


def resolve_data_root(p: experimentParams) -> Path:
    """
    Expand and resolve the absolute path to data_root
    """
    return Path(p.data_root).resolve()
