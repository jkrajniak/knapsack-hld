"""Phase C §3.4: structural and cross-baseline tests for trs2008.

The §5 worked example from the EJOR paper is not transcribed here
(closed-access source). See `test_trs2008_example.py` for the
placeholder pinned to that task. The tests below validate

1. registration in the unified solver registry,
2. Phase A: champion items are placed in best-ratio-first order,
3. Phase B: profit-greedy fills only deferred classes,
4. Selective-MCKP semantics (a class may be skipped entirely),
5. feasibility on randomly generated instances across all correlation
   classes, and
6. that the heuristic profit never exceeds the MILP optimum on small
   instances (correctness of the upper bound).
"""

from __future__ import annotations

import pytest
from instances.schema import CorrelationKind, InstanceModel
from solvers import SolverStatus, get_solver, list_solvers, validate_solution

from instances import generate_instance


def test_registered_in_solver_registry() -> None:
    """trs2008 must be importable through the unified registry."""
    assert "trs2008" in list_solvers()
    solver = get_solver("trs2008")
    assert solver.name == "trs2008"


def test_phase_a_selects_classes_in_best_ratio_order() -> None:
    """Three classes with disjoint champion ratios — Phase A picks all of them.

    Class 0 champion: (p=10, c=1) → ratio 10.0
    Class 1 champion: (p=8,  c=2) → ratio 4.0
    Class 2 champion: (p=6,  c=3) → ratio 2.0
    Budget 6 fits all three champions exactly (1+2+3=6).
    """
    inst = InstanceModel(
        N=3,
        M=2,
        correlation=CorrelationKind.UNCORRELATED,
        f=0.5,
        seed=0,
        B=6,
        items=[
            [[10, 1], [3, 5]],
            [[8, 2], [4, 6]],
            [[6, 3], [2, 7]],
        ],
    )
    result = get_solver("trs2008").solve(inst)
    validate_solution(inst, result)
    assert result.items_selected == {0: 0, 1: 0, 2: 0}
    assert result.profit == 24
    assert result.total_cost == 6
    assert result.solver_metadata["phase_a_classes_selected"] == 3
    assert result.solver_metadata["phase_b_classes_deferred"] == 0


def test_phase_b_fills_deferred_class_with_lower_profit_item() -> None:
    """A class whose champion does not fit must be revisited in Phase B.

    Class 0 champion: (p=20, c=2) — fits, ratio 10.
    Class 1 champion: (p=30, c=10) — ratio 3, does NOT fit (budget too tight).
    Class 1 also has (p=8, c=4) which DOES fit and Phase B picks up.
    Budget 6: Phase A places (20, 2); Phase B has 4 left and chooses (8, 4).
    """
    inst = InstanceModel(
        N=2,
        M=2,
        correlation=CorrelationKind.UNCORRELATED,
        f=0.5,
        seed=0,
        B=6,
        items=[
            [[20, 2], [10, 5]],
            [[30, 10], [8, 4]],
        ],
    )
    result = get_solver("trs2008").solve(inst)
    validate_solution(inst, result)
    assert result.items_selected == {0: 0, 1: 1}
    assert result.profit == 28
    assert result.total_cost == 6
    assert result.solver_metadata["phase_a_classes_selected"] == 1
    assert result.solver_metadata["phase_b_classes_deferred"] == 1
    assert result.solver_metadata["phase_b_profit"] == 8


def test_class_skipped_when_no_item_fits() -> None:
    """Selective-MCKP: a class with all items too expensive must be skipped."""
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
    result = get_solver("trs2008").solve(inst)
    validate_solution(inst, result)
    assert result.items_selected[1] is None
    assert result.items_selected[0] in {0, 1}
    assert result.profit in {10, 12}
    assert result.status is SolverStatus.FEASIBLE


def test_zero_profit_items_are_ignored() -> None:
    """Items with p == 0 must never be selected (no objective contribution)."""
    inst = InstanceModel(
        N=1,
        M=2,
        correlation=CorrelationKind.UNCORRELATED,
        f=0.5,
        seed=0,
        B=10,
        items=[[[0, 1], [5, 2]]],
    )
    result = get_solver("trs2008").solve(inst)
    validate_solution(inst, result)
    assert result.items_selected == {0: 1}
    assert result.profit == 5


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
def test_trs2008_is_feasible_and_dominated_by_optimum(
    correlation: CorrelationKind, seed: int
) -> None:
    """On small generated instances, trs2008 must be feasible and ≤ HiGHS optimum."""
    inst = generate_instance(N=10, M=4, correlation=correlation, f=0.5, seed=seed)
    heur = get_solver("trs2008").solve(inst)
    opt = get_solver("highs").solve(inst)
    validate_solution(inst, heur)
    validate_solution(inst, opt)
    assert heur.status is SolverStatus.FEASIBLE
    assert opt.status is SolverStatus.OPTIMAL
    assert heur.profit <= opt.profit, (
        f"trs2008 profit {heur.profit} exceeded optimum {opt.profit} on {correlation} seed={seed}"
    )


def test_metadata_phase_split_sums_to_total() -> None:
    """Phase-A + Phase-B profit must equal total profit reported."""
    inst = generate_instance(N=20, M=5, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=42)
    result = get_solver("trs2008").solve(inst)
    meta = result.solver_metadata
    assert meta["phase_a_profit"] + meta["phase_b_profit"] == result.profit
    assert meta["phase_a_cost"] + meta["phase_b_cost"] == result.total_cost
