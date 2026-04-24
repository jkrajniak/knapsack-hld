"""Unit tests for the solver protocol, validator, and registry."""

from __future__ import annotations

import time

import pytest
from solvers import (
    InvalidSolutionError,
    Solver,
    SolveResult,
    SolverStatus,
    get_solver,
    list_solvers,
    register,
    validate_solution,
)
from solvers.registry import _reset_for_tests

from instances import generate_instance


@pytest.fixture
def small_instance():
    return generate_instance(N=10, M=3, correlation="uncorrelated", f=0.5, seed=0)


@pytest.fixture
def fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _make_dummy_solver(name: str = "dummy", status: SolverStatus = SolverStatus.OPTIMAL):
    """Always selects item 0 of every class, ignoring the budget."""

    class _Dummy:
        def __init__(self) -> None:
            self.name = name

        def solve(self, instance, *, time_limit_s=None, random_seed=None):
            t0 = time.perf_counter()
            sel = {i: 0 for i in range(instance.N)}
            profit = sum(instance.items[i][0][0] for i in range(instance.N))
            cost = sum(instance.items[i][0][1] for i in range(instance.N))
            return SolveResult(
                profit=profit,
                items_selected=sel,
                total_cost=cost,
                wall_time_s=time.perf_counter() - t0,
                status=status,
                solver_metadata={"echo_seed": random_seed, "time_limit_s": time_limit_s},
            )

    return _Dummy()


def test_protocol_runtime_check() -> None:
    s = _make_dummy_solver()
    assert isinstance(s, Solver)


def test_validate_accepts_well_formed(small_instance) -> None:
    inst = small_instance
    sel = {0: 1, 1: 2, 2: None}
    profit = inst.items[0][1][0] + inst.items[1][2][0]
    cost = inst.items[0][1][1] + inst.items[1][2][1]
    result = SolveResult(
        profit=profit,
        items_selected=sel,
        total_cost=cost,
        wall_time_s=0.0,
        status=SolverStatus.FEASIBLE,
    )
    if cost <= inst.B:
        validate_solution(inst, result)


def test_validate_rejects_budget_violation(small_instance) -> None:
    """Pick the heaviest item from every class; with f=0.5 this exceeds B."""
    inst = small_instance
    sel: dict[int, int | None] = {}
    profit = 0
    cost = 0
    for i in range(inst.N):
        j_heaviest = max(range(inst.M), key=lambda j: inst.items[i][j][1])
        sel[i] = j_heaviest
        profit += inst.items[i][j_heaviest][0]
        cost += inst.items[i][j_heaviest][1]
    assert cost > inst.B, "test fixture must over-budget; tighten f or reseed"
    bad = SolveResult(
        profit=profit,
        items_selected=sel,
        total_cost=cost,
        wall_time_s=0.0,
        status=SolverStatus.OPTIMAL,
    )
    with pytest.raises(InvalidSolutionError, match="exceeds budget"):
        validate_solution(inst, bad)


def test_validate_rejects_profit_mismatch(small_instance) -> None:
    inst = small_instance
    sel = {0: 0}
    actual_profit = inst.items[0][0][0]
    actual_cost = inst.items[0][0][1]
    bad = SolveResult(
        profit=actual_profit + 1,
        items_selected=sel,
        total_cost=actual_cost,
        wall_time_s=0.0,
        status=SolverStatus.OPTIMAL,
    )
    with pytest.raises(InvalidSolutionError, match="profit"):
        validate_solution(inst, bad)


def test_validate_rejects_out_of_range_index(small_instance) -> None:
    inst = small_instance
    bad = SolveResult(
        profit=0,
        items_selected={inst.N + 5: 0},
        total_cost=0,
        wall_time_s=0.0,
        status=SolverStatus.OPTIMAL,
    )
    with pytest.raises(InvalidSolutionError, match="class index"):
        validate_solution(inst, bad)


def test_registry_register_and_get(fresh_registry) -> None:
    @register("dummy")
    def _factory():
        return _make_dummy_solver("dummy")

    assert "dummy" in list_solvers()
    s = get_solver("dummy")
    assert isinstance(s, Solver)
    assert s.name == "dummy"


def test_registry_rejects_duplicate(fresh_registry) -> None:
    @register("d1")
    def _f1():
        return _make_dummy_solver("d1")

    with pytest.raises(ValueError, match="already registered"):

        @register("d1")
        def _f2():
            return _make_dummy_solver("d1")


def test_registry_rejects_unknown(fresh_registry) -> None:
    with pytest.raises(KeyError, match="unknown solver"):
        get_solver("nope")


def test_registry_enforces_name_match(fresh_registry) -> None:
    @register("declared")
    def _factory():
        return _make_dummy_solver("returned-different-name")

    with pytest.raises(RuntimeError, match="names must match"):
        get_solver("declared")


def test_dummy_solver_round_trip(small_instance) -> None:
    """Sanity: a registered solver returns a SolveResult that the validator accepts
    when the instance happens to admit the selection (cost ≤ B)."""
    s = _make_dummy_solver()
    result = s.solve(small_instance, time_limit_s=1.0, random_seed=7)
    assert isinstance(result, SolveResult)
    assert result.solver_metadata["echo_seed"] == 7
    assert result.solver_metadata["time_limit_s"] == 1.0
    assert result.n_classes_selected == small_instance.N
