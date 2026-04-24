"""Phase C §3.2: HiGHS, SCIP, and CBC must agree on the optimum.

Regression test against the manuscript's claim that the three open-source
MILP baselines are interchangeable for the small/medium scale used as
ground truth in tuning and §4 sanity checks.
"""

from __future__ import annotations

import pytest
from instances.schema import CorrelationKind
from solvers import SolverStatus, get_solver, list_solvers, validate_solution

from instances import generate_instance

MILP_NAMES = ("highs", "scip", "cbc")


def test_milp_adapters_are_registered() -> None:
    assert set(MILP_NAMES) <= set(list_solvers())


@pytest.mark.parametrize(
    "correlation",
    [
        CorrelationKind.UNCORRELATED,
        CorrelationKind.WEAKLY,
        CorrelationKind.STRONGLY,
        CorrelationKind.INVERSELY_STRONGLY,
    ],
)
def test_three_milp_solvers_agree_on_50_class_instance(correlation: CorrelationKind) -> None:
    """Spec gate: HiGHS/SCIP/CBC return the same optimum on every correlation class."""
    inst = generate_instance(N=50, M=5, correlation=correlation, f=0.5, seed=0)

    profits: dict[str, int] = {}
    for name in MILP_NAMES:
        result = get_solver(name).solve(inst, time_limit_s=60.0, random_seed=0)
        validate_solution(inst, result)
        assert result.status is SolverStatus.OPTIMAL, (
            f"{name} returned status {result.status}; expected OPTIMAL on a 50-class instance"
        )
        profits[name] = result.profit

    unique = set(profits.values())
    assert len(unique) == 1, (
        f"MILP baselines disagree on {correlation}: {profits}"
    )


def test_solver_metadata_carries_status_and_objective() -> None:
    """Every adapter must surface its native status string and objective value."""
    inst = generate_instance(N=20, M=3, correlation="uncorrelated", f=0.5, seed=1)
    for name in MILP_NAMES:
        r = get_solver(name).solve(inst, time_limit_s=30.0)
        meta = r.solver_metadata
        assert any(k.endswith("_status") for k in meta), (
            f"{name} metadata missing native status field; got keys {list(meta)}"
        )
        assert "objective_value" in meta
        assert abs(meta["objective_value"] - r.profit) < 0.5
