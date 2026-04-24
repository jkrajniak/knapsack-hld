"""HLD solver: structural correctness, metadata, and instance-dependent lambda_max."""

from __future__ import annotations

import itertools
import math

import pytest
from instances.generator import generate_instance
from instances.schema import CorrelationKind
from solvers import get_solver, validate_solution
from solvers.hld import (
    DEFAULT_ALPHA,
    DEFAULT_K,
    DEFAULT_N_ITER,
    HldAdapter,
    _instance_dependent_lambda_max,
    _split_classes,
)


def test_hld_is_registered() -> None:
    assert get_solver("hld").name == "hld"


def test_default_parameters_match_manuscript() -> None:
    """N_iter=20, alpha=0.9, K=8 are the values reported in §2.7 of the manuscript."""
    solver = HldAdapter()
    assert solver.n_iter == DEFAULT_N_ITER == 20
    assert solver.alpha == DEFAULT_ALPHA == 0.9
    assert solver.k == DEFAULT_K == 8


def test_lambda_max_is_instance_dependent() -> None:
    """`lambda_max = ceil(max p_ij / c_ij)`, not a hard-coded 10."""
    inst = generate_instance(N=6, M=3, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=11)
    expected = math.ceil(max(p / c for cls in inst.items for (p, c) in cls if c > 0))
    assert _instance_dependent_lambda_max(inst) == expected


def test_lambda_max_handles_zero_cost_items() -> None:
    """`c_ij == 0` items must not crash lambda_max."""
    inst = generate_instance(N=4, M=3, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=12)
    items = [list(cls) for cls in inst.items]
    items[0][0] = [50, 0]
    inst2 = inst.model_copy(update={"items": items})
    val = _instance_dependent_lambda_max(inst2)
    assert val >= 1.0
    assert math.isfinite(val)


def test_phase1_trajectory_has_n_iter_entries() -> None:
    inst = generate_instance(N=8, M=3, correlation=CorrelationKind.WEAKLY, f=0.5, seed=21)
    solver = HldAdapter(n_iter=15)
    res = solver.solve(inst, time_limit_s=20.0)
    traj = res.solver_metadata["phase1_trajectory"]
    assert len(traj) == 15
    keys = {"iter", "lambda_lo", "lambda_mid", "lambda_hi", "total_cost"}
    for entry in traj:
        assert keys == set(entry.keys())
    for prev, nxt in itertools.pairwise(traj):
        assert nxt["iter"] == prev["iter"] + 1


def test_phase2_allocation_sums_within_budget() -> None:
    inst = generate_instance(N=16, M=4, correlation=CorrelationKind.WEAKLY, f=0.5, seed=22)
    res = HldAdapter(k=4).solve(inst, time_limit_s=20.0)
    alloc = res.solver_metadata["phase2_allocation"]
    assert len(alloc) == 4
    total_alloc = sum(b["B_k"] for b in alloc)
    assert total_alloc <= inst.B + 4


def test_phase3_logs_one_entry_per_batch() -> None:
    inst = generate_instance(N=12, M=3, correlation=CorrelationKind.STRONGLY, f=0.5, seed=23)
    res = HldAdapter(k=6).solve(inst, time_limit_s=20.0)
    batches = res.solver_metadata["phase3_batches"]
    assert len(batches) == 6
    keys = {"batch", "B_k", "n_items", "sub_milp_wall_s", "profit", "cost", "status"}
    for entry in batches:
        assert keys == set(entry.keys())
        assert entry["sub_milp_wall_s"] >= 0.0


def test_solution_is_feasible_and_dominated_by_exact() -> None:
    inst = generate_instance(N=20, M=4, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=31)
    hld_res = get_solver("hld").solve(inst, time_limit_s=20.0)
    exact = get_solver("highs").solve(inst, time_limit_s=20.0)

    validate_solution(inst, hld_res)
    assert hld_res.total_cost <= inst.B
    assert hld_res.profit <= exact.profit


def test_k_equal_to_one_recovers_global_optimum() -> None:
    """K=1 collapses HLD to a single global MILP solve."""
    inst = generate_instance(N=10, M=3, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=32)
    hld_res = HldAdapter(k=1).solve(inst, time_limit_s=20.0)
    exact = get_solver("highs").solve(inst, time_limit_s=20.0)
    assert hld_res.profit == exact.profit


def test_split_classes_is_balanced_and_total() -> None:
    parts = _split_classes(10, 3)
    sizes = [len(p) for p in parts]
    assert sum(sizes) == 10
    assert max(sizes) - min(sizes) <= 1
    flat = [c for p in parts for c in p]
    assert flat == list(range(10))


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError):
        HldAdapter(n_iter=0)
    with pytest.raises(ValueError):
        HldAdapter(alpha=1.5)
    with pytest.raises(ValueError):
        HldAdapter(k=0)


def test_alpha_zero_falls_back_to_equal_split_when_lambda_uninformative() -> None:
    """alpha=0 means proportional component is silenced; B_k = B/K exactly."""
    inst = generate_instance(N=8, M=3, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=41)
    res = HldAdapter(alpha=0.0, k=4).solve(inst, time_limit_s=20.0)
    alloc = res.solver_metadata["phase2_allocation"]
    expected = inst.B / 4
    for b in alloc:
        assert abs(b["B_k"] - expected) <= 1


def test_sub_solver_is_swappable() -> None:
    """HLD must accept any registered exact solver as its sub-solver."""
    inst = generate_instance(N=8, M=3, correlation=CorrelationKind.WEAKLY, f=0.5, seed=51)
    cbc_res = HldAdapter(sub_solver="cbc", k=4).solve(inst, time_limit_s=30.0)
    highs_res = HldAdapter(sub_solver="highs", k=4).solve(inst, time_limit_s=30.0)
    validate_solution(inst, cbc_res)
    validate_solution(inst, highs_res)
    assert cbc_res.profit == highs_res.profit
