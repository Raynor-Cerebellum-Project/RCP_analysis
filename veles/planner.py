from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from RCP_analysis.python.functions.pipeline_hierarchy import (
    COLUMN_DEPENDENCIES,
    get_all_ancestor_columns,
    parse_status_timestamp,
)
from veles.data import Session

Policy = Literal["needed", "all", "none"]

STALE_TOLERANCE_S = 2.0 # If a parent step ran more than this many seconds after a child step, the child is considered stale and will be re-run.


class CellState(Enum):
    RUN = "run"
    MISSING = "missing"
    STALE = "stale"
    FAILED = "failed"
    RERUN = "rerun"
    SKIP = "skip"
    NA = "n/a"
    SESSION_SKIPPED = "session skipped"


# States in which the step will execute during the run.
RAN_STATES = {CellState.RUN, CellState.MISSING, CellState.STALE, CellState.FAILED, CellState.RERUN}

# States the user must confirm on the Review step (§2.8).
WARN_STATES = {CellState.MISSING, CellState.STALE, CellState.FAILED}


@dataclass
class PlanWarning:
    session: str
    step: str
    message: str


@dataclass
class RunRequest:
    animal: str
    conditions: dict[str, list[int]]
    targets: list[str]
    ua_variant: Literal["ssmf", "mf"] = "ssmf"
    policy: Policy = "needed"
    on_failure: Literal["skip_session", "stop_run"] = "skip_session"
    output_root: Path | None = None


@dataclass
class Plan:
    steps: list[str]
    cells: dict[tuple[str, str], CellState]
    warnings: list[PlanWarning] = field(default_factory=list)


def close_and_order_steps(targets: list[str]) -> list[str]:
    unknown = [t for t in targets if t not in COLUMN_DEPENDENCIES]
    if unknown:
        raise ValueError(f"Unknown pipeline step(s): {unknown}")

    needed: set[str] = set(targets)
    for t in targets:
        needed.update(get_all_ancestor_columns(t))

    catalog_index = {name: i for i, name in enumerate(COLUMN_DEPENDENCIES)}
    waiting_on = {n: {d for d in COLUMN_DEPENDENCIES[n] if d in needed} for n in needed}

    ordered: list[str] = []
    ready = [n for n, deps in waiting_on.items() if not deps]
    while ready:
        ready.sort(key=catalog_index.__getitem__)
        node = ready.pop(0)
        ordered.append(node)
        for n, deps in waiting_on.items():
            if node in deps:
                deps.discard(node)
                if not deps:
                    ready.append(n)

    if len(ordered) != len(needed):
        raise ValueError(f"Dependency cycle among: {sorted(needed - set(ordered))}")
    return ordered


def is_applicable(step: str, session: Session) -> bool:
    if step == "VOG":
        return any(c.values.get("VOG_File", "").strip() for c in session.conditions)
    return True


def stale_reasons(step: str, session: Session) -> list[str]:
    reasons: list[str] = []
    step_dt = parse_status_timestamp(session.status.get(step, ""))
    if step_dt is not None:
        for parent in COLUMN_DEPENDENCIES.get(step, []):
            parent_dt = parse_status_timestamp(session.status.get(parent, ""))
            if parent_dt is not None and (parent_dt - step_dt).total_seconds() > STALE_TOLERANCE_S:
                reasons.append(f"'{parent}' ran after '{step}' ({parent_dt:%m/%d/%Y %H:%M:%S})")
    if step in session.stale_file_inputs:
        reasons.append(f"a file input of '{step}' changed after it ran")
    return reasons


def decide_cell_state(
    *,
    is_target: bool,
    is_applicable: bool,
    status_value: str,
    is_stale: bool,
    upstream_ran: bool,
    policy: Policy,
) -> CellState:
    """Pick the §5 rule 2 state for one (session, step) cell.

    status_value: raw data_status_reaching.csv cell ('', 'DONE', 'FAIL', 'IN-PROGRESS' or a timestamp).
    upstream_ran: an ancestor of this step is already planned to execute for this session.
    """
    status = status_value.upper()
    # Decided by Keondre
    if not is_applicable:
        return CellState.NA
    if is_target:
        return CellState.RUN
    if policy == "none":
        return CellState.SKIP
    if status_value == "":
        return CellState.MISSING
    if status == "FAIL":
        return CellState.FAILED
    if status == "IN-PROGRESS":
        return CellState.FAILED 
    if is_stale:
        return CellState.STALE
    if upstream_ran:
        return CellState.RERUN
    if policy == "all":
        return CellState.RERUN  
    return CellState.SKIP

def build_plan(request: RunRequest, sessions: dict[str, Session]) -> Plan:
    steps = close_and_order_steps(request.targets)
    cells: dict[tuple[str, str], CellState] = {}
    warnings: list[PlanWarning] = []

    for name, session in sessions.items():
        if not request.conditions.get(name):
            for step in steps:
                cells[(name, step)] = CellState.SESSION_SKIPPED
            continue

        ran: set[str] = set()
        for step in steps:
            status_value = session.status.get(step, "").strip()
            reasons = stale_reasons(step, session)
            state = decide_cell_state(
                is_target=step in request.targets,
                is_applicable=is_applicable(step, session),
                status_value=status_value,
                is_stale=bool(reasons),
                upstream_ran=any(a in ran for a in get_all_ancestor_columns(step)),
                policy=request.policy,
            )
            cells[(name, step)] = state
            if state in RAN_STATES:
                ran.add(step)

            if state in WARN_STATES:
                detail = "; ".join(reasons) if state is CellState.STALE else f"status is '{status_value or 'empty'}'"
                warnings.append(PlanWarning(name, step, f"{step} is {state.value}: {detail}"))
            if status_value.upper() == "DONE" and request.policy != "none":
                warnings.append(PlanWarning(name, step, f"{step} is DONE with no timestamp; staleness can't be checked"))

    return Plan(steps=steps, cells=cells, warnings=warnings)
