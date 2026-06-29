"""Guarded-HLD: never-lose backstop, Lagrangian-UB skip, and feasibility."""

from __future__ import annotations

import pytest
from instances.generator import generate_instance
from instances.schema import CorrelationKind
from solvers import get_solver, validate_solution
from solvers.guarded_hld import DEFAULT_TAU_SKIP, GuardedHldAdapter, _lagrangian_upper_bound


def test_guarded_hld_is_registered() -> None:
    assert get_solver("guarded_hld").name == "guarded_hld"


def test_invalid_tau_skip_raises() -> None:
    with pytest.raises(ValueError, match="tau_skip"):
        GuardedHldAdapter(tau_skip=-0.1)
    with pytest.raises(ValueError, match="tau_skip"):
        GuardedHldAdapter(tau_skip=1.5)


def test_lagrangian_upper_bound_dominates_optimum() -> None:
    """U(lambda_est) >= exact optimum for any lambda >= 0 (weak duality)."""
    inst = generate_instance(N=14, M=4, correlation=CorrelationKind.WEAKLY, f=0.5, seed=7)
    exact = get_solver("highs").solve(inst, time_limit_s=20.0)
    for lam in (0.0, 0.5, 1.0, 5.0, 10.0):
        ub = _lagrangian_upper_bound(inst, lam)
        assert ub >= exact.profit, f"U({lam})={ub} < optimum {exact.profit}"


def test_guarded_never_loses_to_po_across_regimes() -> None:
    """At matching K, guarded profit >= PO profit on every (correlation, f) regime.

    The guarantee is relative to the equal-split decomposition at the same K the
    guard uses internally (different K gives different splits and is not the
    relevant baseline).
    """
    regimes = [
        (CorrelationKind.INVERSELY_STRONGLY, 0.5),
        (CorrelationKind.INVERSELY_STRONGLY, 0.9),
        (CorrelationKind.STRONGLY, 0.5),
        (CorrelationKind.STRONGLY, 0.9),
    ]
    k = 10
    for corr, f in regimes:
        inst = generate_instance(N=40, M=10, correlation=corr, f=f, seed=3)
        po = get_solver("partition_optimal").__class__(k=k).solve(inst, time_limit_s=20.0)
        guarded = GuardedHldAdapter(k=k, n_iter=20, lambda_max_override=80.745).solve(
            inst, time_limit_s=60.0
        )
        validate_solution(inst, guarded)
        assert guarded.profit >= po.profit, (
            f"guarded {guarded.profit} < PO {po.profit} on {corr}/f={f} at K={k}; "
            f"decision={guarded.solver_metadata['decision']}"
        )


def test_guarded_contains_hld_harm_on_strongly_loose() -> None:
    """The harm regime: strongly / f=0.9. Guarded must not be worse than PO at same K."""
    k = 20
    inst = generate_instance(
        N=60, M=10, correlation=CorrelationKind.STRONGLY, f=0.9, seed=5
    )
    po = get_solver("partition_optimal").__class__(k=k).solve(inst, time_limit_s=20.0)
    guarded = GuardedHldAdapter(k=k, n_iter=25, lambda_max_override=80.745).solve(
        inst, time_limit_s=60.0
    )
    assert guarded.profit >= po.profit
    assert guarded.solver_metadata["decision"] in {"skip", "po_wins", "hld_wins"}


def test_tau_skip_one_always_skips() -> None:
    """tau_skip=1.0 => PO is always within 100% of U_L => always skip."""
    inst = generate_instance(N=12, M=3, correlation=CorrelationKind.WEAKLY, f=0.5, seed=9)
    res = GuardedHldAdapter(k=4, tau_skip=1.0).solve(inst, time_limit_s=20.0)
    assert res.solver_metadata["decision"] == "skip"
    assert res.solver_metadata["hld_profit"] is None
    assert res.solver_metadata["wall_hld_s"] == 0.0


def test_tau_skip_zero_never_skips_and_picks_winner() -> None:
    """tau_skip=0 => never skip; decision is po_wins or hld_wins; profit = max."""
    inst = generate_instance(
        N=40, M=10, correlation=CorrelationKind.INVERSELY_STRONGLY, f=0.5, seed=11
    )
    po = get_solver("partition_optimal").solve(inst, time_limit_s=20.0)
    res = GuardedHldAdapter(k=10, n_iter=20, tau_skip=0.0, lambda_max_override=80.745).solve(
        inst, time_limit_s=60.0
    )
    meta = res.solver_metadata
    assert meta["decision"] in {"po_wins", "hld_wins"}
    assert meta["hld_profit"] is not None
    assert res.profit == max(po.profit, meta["hld_profit"])


def test_metadata_schema() -> None:
    inst = generate_instance(N=10, M=3, correlation=CorrelationKind.WEAKLY, f=0.5, seed=13)
    res = GuardedHldAdapter(k=4, tau_skip=DEFAULT_TAU_SKIP).solve(inst, time_limit_s=20.0)
    keys = {"decision", "po_profit", "hld_profit", "lagrangian_ub", "lambda_est",
            "po_gap_to_ul", "tau_skip", "wall_po_s", "wall_hld_s", "sub"}
    assert keys <= set(res.solver_metadata.keys())
    assert "po" in res.solver_metadata["sub"]
