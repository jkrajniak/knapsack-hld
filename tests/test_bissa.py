"""BISSA correctness + Selective-MCKP transformation audit."""

from __future__ import annotations

import pytest
from heuristics.bissa import _augment_with_dummy
from instances.generator import generate_instance
from instances.schema import CorrelationKind
from solvers import get_solver, validate_solution


def test_bissa_is_registered() -> None:
    solver = get_solver("bissa")
    assert solver.name == "bissa"


def test_bissa_records_transformation_in_metadata() -> None:
    inst = generate_instance(N=10, M=4, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=11)
    res = get_solver("bissa").solve(inst)
    assert res.solver_metadata["transformation"] == "mckp_to_selective_mckp_dummy_item"


def test_augment_with_dummy_prepends_zero_zero_per_class() -> None:
    items = [[[5, 3], [7, 4]], [[2, 1]]]
    augmented = _augment_with_dummy(items)
    assert len(augmented) == len(items)
    for cls in augmented:
        assert cls[0] == (0, 0)
    assert augmented[0][1:] == [(5, 3), (7, 4)]
    assert augmented[1][1:] == [(2, 1)]


@pytest.mark.parametrize(
    "correlation",
    [
        CorrelationKind.UNCORRELATED,
        CorrelationKind.WEAKLY,
        CorrelationKind.STRONGLY,
        CorrelationKind.INVERSELY_STRONGLY,
    ],
)
def test_bissa_returns_feasible_solution(correlation: CorrelationKind) -> None:
    inst = generate_instance(N=15, M=4, correlation=correlation, f=0.5, seed=7)
    res = get_solver("bissa").solve(inst, time_limit_s=10.0)
    validate_solution(inst, res)
    assert res.total_cost <= inst.B


def test_bissa_dominated_by_exact() -> None:
    inst = generate_instance(N=20, M=4, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=21)
    bissa_res = get_solver("bissa").solve(inst, time_limit_s=10.0)
    exact_res = get_solver("highs").solve(inst, time_limit_s=10.0)
    assert bissa_res.profit <= exact_res.profit


def test_bissa_competitive_with_greedy_on_loose_budgets() -> None:
    inst = generate_instance(N=20, M=4, correlation=CorrelationKind.WEAKLY, f=0.9, seed=33)
    bissa_res = get_solver("bissa").solve(inst, time_limit_s=10.0)
    greedy_res = get_solver("greedy_max_ratio").solve(inst)
    assert bissa_res.profit >= int(0.6 * greedy_res.profit)
