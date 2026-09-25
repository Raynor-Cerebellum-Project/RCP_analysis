import pytest

from veles.data import Condition, Session
from veles.planner import (
    CellState,
    RunRequest,
    build_plan,
    close_and_order_steps,
    decide_cell_state,
    is_applicable,
    stale_reasons,
)
from RCP_analysis.python.functions.pipeline_hierarchy import COLUMN_DEPENDENCIES

S = CellState
TS = "09/01/2026 10:00:00"

EXTRACT_PERI_CHAIN = [
    "OCR",
    "DLC",
    "NPRW_Intan_analysis",
    "compute_shifts",
    "UA_BR_analysis",
    "make_aligned",
    "extract_peri",
]


def up_to_date_status() -> dict[str, str]:
    """Every step in the extract_peri chain ran in order, one minute apart."""
    return {step: f"09/01/2026 10:{i:02d}:00" for i, step in enumerate(EXTRACT_PERI_CHAIN)}


def make_session(status: dict[str, str], name: str = "NRR_RW012", **kw) -> Session:
    return Session(
        name=name,
        location="Loc",
        conditions=[Condition(br_file=1, values={}), Condition(br_file=2, values={})],
        status=status,
        **kw,
    )


def plan_for(session: Session, targets: list[str], policy: str = "needed"):
    request = RunRequest(
        animal="Nike",
        conditions={session.name: [1, 2]},
        targets=targets,
        policy=policy,
    )
    plan = build_plan(request, {session.name: session})
    return plan, {step: plan.cells[(session.name, step)] for step in plan.steps}


# --- close_and_order_steps (§5 rule 1) ---

def test_extract_peri_closes_over_its_ancestors_in_catalog_order():
    assert close_and_order_steps(["extract_peri"]) == EXTRACT_PERI_CHAIN


def test_every_step_comes_after_its_parents():
    order = close_and_order_steps(["plot_raster", "plot_lfp", "combine_UA_gifs", "VOG"])
    for step in order:
        for parent in COLUMN_DEPENDENCIES[step]:
            assert order.index(parent) < order.index(step)


def test_order_ignores_target_order():
    assert close_and_order_steps(["plot_lfp", "extract_peri"]) == close_and_order_steps(["extract_peri", "plot_lfp"])


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="Unknown"):
        close_and_order_steps(["not_a_step"])


# --- decide_cell_state (§5 rule 2), unambiguous cases only ---

def decide(**overrides) -> CellState:
    args = dict(is_target=False, is_applicable=True, status_value=TS, is_stale=False, upstream_ran=False, policy="needed")
    args.update(overrides)
    return decide_cell_state(**args)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"is_target": True}, S.RUN),
        ({"status_value": ""}, S.MISSING),
        ({"status_value": "FAIL"}, S.FAILED),
        ({"status_value": "IN-PROGRESS"}, S.FAILED),
        ({"is_stale": True}, S.STALE),
        ({"upstream_ran": True}, S.RERUN),
        ({}, S.SKIP),
        ({"status_value": "DONE"}, S.SKIP),
        ({"is_applicable": False}, S.NA),
        ({"is_applicable": False, "status_value": ""}, S.NA),  # never-run VOG is n/a, not missing
        ({"is_applicable": False, "is_target": True}, S.NA),  # ticking VOG can't run it without a VOG_File
        ({"status_value": "fail"}, S.FAILED),
        ({"policy": "all"}, S.RERUN),
        ({"policy": "none", "status_value": ""}, S.SKIP),
        ({"policy": "none", "is_target": True}, S.RUN),
    ],
)
def test_decide_cell_state(overrides, expected):
    assert decide(**overrides) is expected


# --- staleness and applicability ---

def test_parent_newer_by_more_than_2s_is_stale():
    status = up_to_date_status()
    status["DLC"] = "09/01/2026 10:03:05"  # 5 s after compute_shifts
    assert stale_reasons("compute_shifts", make_session(status))


def test_parent_newer_by_1s_is_within_tolerance():
    status = up_to_date_status()
    status["DLC"] = "09/01/2026 10:03:01"
    assert stale_reasons("compute_shifts", make_session(status)) == []


def test_vog_needs_a_vog_file():
    without = make_session({})
    with_vog = Session("S", "Loc", [Condition(1, {"VOG_File": "vog_01.avi"})])
    assert not is_applicable("VOG", without)
    assert is_applicable("VOG", with_vog)
    assert is_applicable("DLC", without)


# --- build_plan ---

def test_everything_up_to_date_only_runs_the_target():
    plan, cells = plan_for(make_session(up_to_date_status()), ["extract_peri"])
    assert cells == {step: S.SKIP for step in EXTRACT_PERI_CHAIN[:-1]} | {"extract_peri": S.RUN}
    assert plan.warnings == []


def test_missing_step_cascades_to_its_descendants_only():
    status = up_to_date_status()
    status["DLC"] = ""
    plan, cells = plan_for(make_session(status), ["extract_peri"])
    assert cells == {
        "OCR": S.SKIP,
        "DLC": S.MISSING,
        "NPRW_Intan_analysis": S.SKIP,  # not downstream of DLC
        "compute_shifts": S.RERUN,
        "UA_BR_analysis": S.RERUN,
        "make_aligned": S.RERUN,
        "extract_peri": S.RUN,
    }
    assert [(w.step, "missing" in w.message) for w in plan.warnings] == [("DLC", True)]


def test_stale_step_warns_with_the_reason():
    status = up_to_date_status()
    status["DLC"] = "09/01/2026 10:03:05"
    plan, cells = plan_for(make_session(status), ["extract_peri"])
    assert cells["compute_shifts"] is S.STALE
    assert cells["UA_BR_analysis"] is S.RERUN
    assert len(plan.warnings) == 1 and "'DLC' ran after" in plan.warnings[0].message


def test_manual_trial_remove_edit_makes_extract_peri_stale():
    status = up_to_date_status() | {"plot_raster": "09/01/2026 10:30:00"}
    session = make_session(status, stale_file_inputs={"extract_peri"})
    plan, cells = plan_for(session, ["plot_raster"])
    assert cells["extract_peri"] is S.STALE
    assert cells["plot_raster"] is S.RUN
    assert "file input" in plan.warnings[0].message


def test_done_without_timestamp_is_up_to_date_but_warns():
    status = up_to_date_status()
    status["make_aligned"] = "DONE"
    plan, cells = plan_for(make_session(status), ["extract_peri"])
    assert cells["make_aligned"] is S.SKIP
    assert [w.step for w in plan.warnings] == ["make_aligned"]


def test_session_with_no_conditions_is_skipped():
    session = make_session(up_to_date_status())
    request = RunRequest(animal="Nike", conditions={session.name: []}, targets=["extract_peri"])
    plan = build_plan(request, {session.name: session})
    assert {plan.cells[(session.name, s)] for s in plan.steps} == {S.SESSION_SKIPPED}


def test_policy_all_reruns_every_prerequisite():
    plan, cells = plan_for(make_session(up_to_date_status()), ["extract_peri"], policy="all")
    assert cells == {step: S.RERUN for step in EXTRACT_PERI_CHAIN[:-1]} | {"extract_peri": S.RUN}
    assert plan.warnings == []


def test_policy_none_runs_targets_without_checks():
    status = up_to_date_status()
    status["DLC"] = ""
    plan, cells = plan_for(make_session(status), ["extract_peri"], policy="none")
    assert cells == {step: S.SKIP for step in EXTRACT_PERI_CHAIN[:-1]} | {"extract_peri": S.RUN}
    assert plan.warnings == []
