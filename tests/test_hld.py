"""HLD solver: structural correctness, metadata, and instance-dependent lambda_max."""

from __future__ import annotations

import itertools
import math

import pytest
from instances.generator import generate_instance
from instances.schema import CorrelationKind
from solvers import get_solver, validate_solution
from solvers.hld import (
    CLASS_ORDERINGS,
    DEFAULT_ALPHA,
    DEFAULT_CLASS_ORDERING,
    DEFAULT_K,
    DEFAULT_N_ITER,
    HldAdapter,
    _class_order,
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


# ---------------------------------------------------------------------------
# class_ordering (Task 3.3.1 of revision-finalization-2026)
# ---------------------------------------------------------------------------


def test_default_class_ordering_is_sequential() -> None:
    """Default keeps the legacy sequential behaviour (manuscript §3.6)."""
    solver = HldAdapter()
    assert solver.class_ordering == DEFAULT_CLASS_ORDERING == "sequential"
    assert set(CLASS_ORDERINGS) == {"sequential", "random", "adversarial"}


def test_class_ordering_invalid_raises() -> None:
    with pytest.raises(ValueError, match="class_ordering"):
        HldAdapter(class_ordering="not-a-real-ordering")


def test_class_order_sequential_is_identity() -> None:
    inst = generate_instance(N=12, M=3, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=61)
    assert _class_order(inst, ordering="sequential", random_seed=None) == list(range(12))


def test_class_order_random_is_permutation_and_deterministic_with_seed() -> None:
    inst = generate_instance(N=20, M=3, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=62)
    a = _class_order(inst, ordering="random", random_seed=7)
    b = _class_order(inst, ordering="random", random_seed=7)
    c = _class_order(inst, ordering="random", random_seed=8)
    assert sorted(a) == list(range(20))  # is a permutation
    assert a == b  # seed-deterministic
    assert a != list(
        range(20)
    )  # actually shuffled (probabilistic, but with N=20 essentially certain)
    assert a != c  # different seed -> different order


def test_class_order_adversarial_sorts_by_descending_max_pc_ratio() -> None:
    """Adversarial puts the highest profit/cost-ratio class first (stresses equal-split decomposition)."""
    inst = generate_instance(N=15, M=4, correlation=CorrelationKind.WEAKLY, f=0.5, seed=63)
    order = _class_order(inst, ordering="adversarial", random_seed=None)
    assert sorted(order) == list(range(15))

    def max_ratio(i: int) -> float:
        return max(
            (p / c for (p, c) in inst.items[i] if c > 0),
            default=0.0,
        )

    ratios_in_order = [max_ratio(i) for i in order]
    for prev, nxt in itertools.pairwise(ratios_in_order):
        assert prev >= nxt


def test_class_ordering_random_does_not_change_solution_feasibility() -> None:
    inst = generate_instance(N=14, M=3, correlation=CorrelationKind.WEAKLY, f=0.5, seed=64)
    res = HldAdapter(k=4, class_ordering="random").solve(inst, time_limit_s=20.0, random_seed=123)
    validate_solution(inst, res)
    assert res.total_cost <= inst.B
    assert res.solver_metadata["params"]["class_ordering"] == "random"
    # The phase-3 batches must collectively cover every class index exactly once.
    selected = res.items_selected
    assert set(selected.keys()) == set(range(inst.N))


def test_class_ordering_adversarial_records_meta() -> None:
    inst = generate_instance(N=10, M=3, correlation=CorrelationKind.STRONGLY, f=0.5, seed=65)
    res = HldAdapter(k=3, class_ordering="adversarial").solve(inst, time_limit_s=20.0)
    validate_solution(inst, res)
    assert res.solver_metadata["params"]["class_ordering"] == "adversarial"


def test_class_ordering_random_reproducible_end_to_end() -> None:
    """Same `random_seed` -> same HLD solution under random ordering."""
    inst = generate_instance(N=12, M=3, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=66)
    a = HldAdapter(k=3, class_ordering="random").solve(inst, time_limit_s=20.0, random_seed=42)
    b = HldAdapter(k=3, class_ordering="random").solve(inst, time_limit_s=20.0, random_seed=42)
    assert a.profit == b.profit
    assert a.items_selected == b.items_selected
