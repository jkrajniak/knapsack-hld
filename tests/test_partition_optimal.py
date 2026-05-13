"""Partition-Optimal heuristic correctness + metadata."""

from __future__ import annotations

import heuristics.partition_optimal as partition_optimal
import pytest
from heuristics.partition_optimal import PartitionOptimalAdapter
from instances.generator import generate_instance
from instances.schema import CorrelationKind
from solvers import SolveResult, SolverStatus, get_solver, validate_solution


def test_partition_optimal_is_registered() -> None:
    solver = get_solver("partition_optimal")
    assert solver.name == "partition_optimal"


def test_k_equal_to_n_classes_falls_back_to_per_class() -> None:
    """When k == N, every batch is exactly one class."""
    inst = generate_instance(N=8, M=4, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=1)
    solver = PartitionOptimalAdapter(k=inst.N, sub_solver="highs")
    res = solver.solve(inst, time_limit_s=10.0)

    validate_solution(inst, res)
    assert res.solver_metadata["k"] == inst.N
    assert len(res.solver_metadata["batches"]) == inst.N
    for batch in res.solver_metadata["batches"]:
        assert batch["n_classes"] == 1


def test_k_equal_to_one_matches_global_optimum() -> None:
    """A single batch with budget = B is identical to a global solve."""
    inst = generate_instance(N=10, M=3, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=2)
    solver = PartitionOptimalAdapter(k=1, sub_solver="highs")
    partition_res = solver.solve(inst, time_limit_s=10.0)
    global_res = get_solver("highs").solve(inst, time_limit_s=10.0)

    assert partition_res.profit == global_res.profit
    validate_solution(inst, partition_res)


def test_default_k_yields_feasible_solution_dominated_by_exact() -> None:
    inst = generate_instance(N=24, M=4, correlation=CorrelationKind.WEAKLY, f=0.5, seed=3)
    res = get_solver("partition_optimal").solve(inst, time_limit_s=10.0)
    exact = get_solver("highs").solve(inst, time_limit_s=10.0)

    validate_solution(inst, res)
    assert res.profit <= exact.profit
    assert res.total_cost <= inst.B


def test_partition_optimal_records_per_batch_metadata() -> None:
    inst = generate_instance(N=12, M=3, correlation=CorrelationKind.STRONGLY, f=0.5, seed=4)
    res = PartitionOptimalAdapter(k=4, sub_solver="highs").solve(inst, time_limit_s=10.0)

    batches = res.solver_metadata["batches"]
    assert len(batches) == 4
    keys = {"batch", "n_classes", "B_k", "profit", "cost", "status", "sub_milp_wall_s"}
    for batch in batches:
        assert keys.issubset(batch.keys())


def test_partition_optimal_reports_timeout_when_subsolvers_find_no_incumbent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All-zero timeout sub-solves should not be reported as feasible."""

    class TimeoutSolver:
        name = "timeout_subsolver"

        def solve(self, instance, *, time_limit_s=None, random_seed=None) -> SolveResult:
            return SolveResult(
                profit=0,
                items_selected={i: None for i in range(instance.N)},
                total_cost=0,
                wall_time_s=0.01,
                status=SolverStatus.TIMEOUT,
            )

    monkeypatch.setattr(partition_optimal, "get_solver", lambda _name: TimeoutSolver())
    inst = generate_instance(N=8, M=3, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=5)

    res = PartitionOptimalAdapter(k=4, sub_solver="timeout_subsolver").solve(
        inst,
        time_limit_s=1.0,
    )

    validate_solution(inst, res)
    assert res.status is SolverStatus.TIMEOUT
    assert res.profit == 0
    assert res.solver_metadata["sub_status_counts"] == {"timeout": 4}


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        PartitionOptimalAdapter(k=0)
