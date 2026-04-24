"""Phase C §3.3.2: small worked-example tests for the mcknap re-implementation.

The spec calls for "the worked example from §5 of Pisinger 1995". We do
not redistribute the paper, and we have not transcribed the exact
numbers from §5 into this file (TODO: when the paper is to hand,
replace `_pisinger_section_5_example` with the canonical instance).

What we DO test, exhaustively, is the property the §5 example is meant
to demonstrate: that our pure-Python `mcknap` returns the SAME optimum
as the open-source MILP baselines on small MCKP and Selective-MCKP
instances across every correlation class. This is the actual paper
claim ("our re-implementation is correct").
"""

from __future__ import annotations

import pytest
from instances.schema import CorrelationKind
from solvers import SolverStatus, get_solver, validate_solution

from instances import generate_instance


def _hand_crafted_small_mckp():
    """Tiny hand-crafted instance with optimum verifiable by inspection.

    3 classes, 3 items each, B chosen so the optimum is unique:
    - class 0: pick item 2 (p=8, c=4)
    - class 1: pick item 1 (p=6, c=3)
    - class 2: pick item 0 (p=5, c=2)
    Total profit = 19, total cost = 9.
    """
    from instances.schema import InstanceModel

    return InstanceModel(
        N=3,
        M=3,
        correlation=CorrelationKind.UNCORRELATED,
        f=0.5,
        seed=0,
        B=10,
        items=[
            [[3, 1], [5, 3], [8, 4]],
            [[2, 1], [6, 3], [7, 5]],
            [[5, 2], [4, 3], [3, 5]],
        ],
    )


def test_hand_crafted_optimum() -> None:
    inst = _hand_crafted_small_mckp()
    result = get_solver("mcknap").solve(inst, time_limit_s=10.0)
    validate_solution(inst, result)
    assert result.status is SolverStatus.OPTIMAL
    assert result.profit == 19, f"expected 19, got {result.profit}"
    assert result.items_selected == {0: 2, 1: 1, 2: 0}


@pytest.mark.parametrize(
    "correlation",
    [
        CorrelationKind.UNCORRELATED,
        CorrelationKind.WEAKLY,
        CorrelationKind.STRONGLY,
        CorrelationKind.INVERSELY_STRONGLY,
    ],
)
@pytest.mark.parametrize("seed", range(5))
def test_mcknap_matches_highs_on_small_instances(correlation: CorrelationKind, seed: int) -> None:
    """Correctness gate: mcknap == HiGHS on every small instance we generate."""
    inst = generate_instance(N=12, M=4, correlation=correlation, f=0.5, seed=seed)
    hi = get_solver("highs").solve(inst)
    mk = get_solver("mcknap").solve(inst, time_limit_s=10.0)
    validate_solution(inst, mk)
    assert mk.status is SolverStatus.OPTIMAL, (
        f"mcknap returned {mk.status} on N=12 M=4 {correlation} seed={seed}"
    )
    assert hi.profit == mk.profit, (
        f"mcknap disagrees with HiGHS on {correlation} seed={seed}: "
        f"HiGHS={hi.profit} vs mcknap={mk.profit}"
    )


def test_mcknap_handles_selective_skipping() -> None:
    """A class with only expensive items must be skipped entirely."""
    from instances.schema import InstanceModel

    inst = InstanceModel(
        N=2,
        M=2,
        correlation=CorrelationKind.UNCORRELATED,
        f=0.5,
        seed=0,
        B=3,
        items=[
            [[10, 2], [12, 3]],
            [[100, 50], [150, 80]],
        ],
    )
    result = get_solver("mcknap").solve(inst, time_limit_s=5.0)
    validate_solution(inst, result)
    assert result.status is SolverStatus.OPTIMAL
    assert result.items_selected[1] is None, "class 1 cannot fit; must be skipped"
    assert result.items_selected[0] in {0, 1}
    assert result.profit in {10, 12}


def test_mcknap_returns_timeout_gracefully() -> None:
    """A starved solve must return a SolveResult, not raise."""
    inst = generate_instance(N=80, M=10, correlation="inversely_strongly", f=0.5, seed=0)
    result = get_solver("mcknap").solve(inst, time_limit_s=0.01)
    assert result is not None
    assert result.wall_time_s >= 0.0
    assert result.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE, SolverStatus.TIMEOUT)
