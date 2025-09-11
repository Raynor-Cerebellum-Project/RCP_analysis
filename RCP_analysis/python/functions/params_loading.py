from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
from scipy.io import loadmat
import h5py


# ---------------- Params model ----------------
@dataclass
class experimentParams:
    data_root: str
    blackrock_rel: str
    geom_mat_rel: str
    highpass_hz: float
    probes: Dict[str, Dict[str, Any]]
    probe_arrays: Dict[str, Dict[str, Dict[str, Any]]]
    parallel_jobs: int
    threads_per_worker: int
    chunk: str
    win_pre_s: float
    win_post_s: float
    stim_bar_ms: float
    dig_line: Optional[str]
    quicklook_rows: int
    quicklook_cols: int
    stride: int
    default_stim_num: int
    sessions: Dict[str, Dict[str, Any]]
    stim_nums: Dict[str, int] = None


def load_experiment_params(yaml_path: Path, repo_root: Path) -> experimentParams:
    """
    Load YAML configuration into a experimentParams dataclass and expand {REPO_ROOT}
    placeholders anywhere they appear (top-level and nested dicts).
    """
    cfg = yaml.safe_load(yaml_path.read_text()) or {}

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
        geom_mat_rel=cfg["geom_mat_rel"],
        highpass_hz=float(cfg["highpass_hz"]),
        probes=cfg.get("probes", {}),
        probe_arrays=cfg.get("probe_arrays", {}),
        parallel_jobs=int(cfg["parallel_jobs"]),
        threads_per_worker=int(cfg["threads_per_worker"]),
        chunk=str(cfg["chunk"]),
        win_pre_s=float(cfg["win_pre_s"]),
        win_post_s=float(cfg["win_post_s"]),
        stim_bar_ms=float(cfg["stim_bar_ms"]),
        dig_line=cfg.get("dig_line") or None,
        quicklook_rows=int(cfg["quicklook_rows"]),
        quicklook_cols=int(cfg["quicklook_cols"]),
        stride=int(cfg["stride"]),
        default_stim_num=int(cfg["default_stim_num"]),
        sessions=cfg.get("sessions", {}) or {},
        stim_nums=cfg.get("stim_nums", {}) or {},
    )


def resolve_data_root(p: experimentParams) -> Path:
    return Path(p.data_root).resolve()
