"""Greedy-MaxRatio sanity tests."""

from __future__ import annotations

import pytest
from instances.generator import generate_instance
from instances.schema import CorrelationKind, InstanceModel
from solvers import get_solver, validate_solution


def _toy_instance(items: list[list[list[int]]], B: int) -> InstanceModel:
    return InstanceModel(
        N=len(items),
        M=len(items[0]),
        correlation=CorrelationKind.UNCORRELATED,
        f=0.5,
        seed=0,
        B=B,
        items=items,
    )


def test_greedy_max_ratio_is_registered() -> None:
    solver = get_solver("greedy_max_ratio")
    assert solver.name == "greedy_max_ratio"


def test_greedy_picks_highest_ratio_first() -> None:
    items = [
        [[10, 10], [5, 1]],
        [[20, 20], [4, 8]],
    ]
    inst = _toy_instance(items, B=20)

    solver = get_solver("greedy_max_ratio")
    res = solver.solve(inst)

    assert res.items_selected[0] == 1
    assert res.items_selected[1] == 1
    assert res.profit == 9
    assert res.total_cost == 9
    validate_solution(inst, res)


def test_greedy_skips_class_when_nothing_fits() -> None:
    items = [
        [[100, 5]],
        [[50, 100]],
    ]
    inst = _toy_instance(items, B=10)

    solver = get_solver("greedy_max_ratio")
    res = solver.solve(inst)

    assert res.items_selected[0] == 0
    assert res.items_selected[1] is None
    assert res.profit == 100
    assert res.total_cost == 5
    validate_solution(inst, res)


@pytest.mark.parametrize(
    "correlation",
    [
        CorrelationKind.UNCORRELATED,
        CorrelationKind.WEAKLY,
        CorrelationKind.STRONGLY,
        CorrelationKind.INVERSELY_STRONGLY,
    ],
)
def test_greedy_is_feasible_on_random_instances(correlation: CorrelationKind) -> None:
    inst = generate_instance(N=20, M=4, correlation=correlation, f=0.5, seed=11)
    res = get_solver("greedy_max_ratio").solve(inst)
    validate_solution(inst, res)
    assert 0 <= res.total_cost <= inst.B


def test_greedy_is_dominated_by_optimal() -> None:
    inst = generate_instance(N=20, M=4, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=42)
    greedy = get_solver("greedy_max_ratio").solve(inst)
    exact = get_solver("highs").solve(inst, time_limit_s=10.0)
    assert greedy.profit <= exact.profit
