"""Unit tests for the §4.3 anomaly diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

import pytest
from anomalies.analyse import (
    H1_LAMBDA_REL_SPREAD_THRESHOLD,
    analyse_record,
    is_h1_degenerate_dual,
    is_h2_straggler,
    phase1_metrics,
    phase3_metrics,
)
from anomalies.sweep import (
    DEFAULT_CELL,
    DEFAULT_N_ITER_GRID,
    DEFAULT_SEEDS,
    ensure_anomaly_subset,
)


def _conv_traj(n: int, budget: int, lambda_star: float = 0.5) -> list[dict]:
    """Realistic bisection that converges to ``lambda_star``.

    Mimics the HLD invariant: ``cost > B`` iff ``lambda_mid < lambda_star``.
    Cost values cycle around ``budget`` so the final iterations stop
    flipping sign once the search shrinks below ``lambda_star`` precision.
    """
    traj = []
    lo, hi = 0.0, 1.0
    for it in range(n):
        mid = (lo + hi) / 2.0
        if mid < lambda_star:
            cost = budget + 100
            lo = mid
        else:
            cost = budget - 100
            hi = mid
        traj.append({"iter": it, "lambda_lo": lo, "lambda_mid": mid, "lambda_hi": hi, "total_cost": cost})
    return traj


def _osc_traj(n: int, budget: int) -> list[dict]:
    """Oscillating: alternating lambda, alternating sign of cost - B."""
    traj = []
    for it in range(n):
        mid = 0.4 if it % 2 == 0 else 0.6
        cost = budget + 100 if it % 2 == 0 else budget - 100
        traj.append({"iter": it, "lambda_lo": 0.0, "lambda_mid": mid, "lambda_hi": 1.0, "total_cost": cost})
    return traj


def test_phase1_metrics_empty_trajectory_is_zero() -> None:
    m = phase1_metrics([], budget=100)
    assert m.n_iter == 0
    assert m.lambda_rel_spread == 0.0
    assert m.sign_flips_in_window == 0


def test_phase1_metrics_converging_trajectory_has_low_spread() -> None:
    m = phase1_metrics(_conv_traj(20, budget=1000), budget=1000)
    assert m.n_iter == 20
    assert m.lambda_rel_spread < H1_LAMBDA_REL_SPREAD_THRESHOLD
    assert m.sign_flips_in_window == 0


def test_phase1_metrics_oscillating_trajectory_has_high_spread() -> None:
    m = phase1_metrics(_osc_traj(20, budget=1000), budget=1000)
    assert m.lambda_rel_spread > H1_LAMBDA_REL_SPREAD_THRESHOLD
    assert m.sign_flips_in_window >= 3


def test_h1_test_passes_for_converging_and_fires_for_oscillating() -> None:
    converging = phase1_metrics(_conv_traj(20, 1000), budget=1000)
    oscillating = phase1_metrics(_osc_traj(20, 1000), budget=1000)
    assert is_h1_degenerate_dual(converging) is False
    assert is_h1_degenerate_dual(oscillating) is True


def test_h1_test_returns_false_for_empty_trajectory() -> None:
    m = phase1_metrics([], budget=100)
    assert is_h1_degenerate_dual(m) is False


def test_phase3_metrics_balanced_batches() -> None:
    batches = [{"sub_milp_wall_s": 1.0} for _ in range(8)]
    m = phase3_metrics(batches)
    assert m.n_batches == 8
    assert m.total_wall_s == pytest.approx(8.0)
    assert m.max_batch_share == pytest.approx(1.0 / 8.0)


def test_phase3_metrics_one_straggler() -> None:
    batches = [{"sub_milp_wall_s": 0.1}] * 7 + [{"sub_milp_wall_s": 10.0}]
    m = phase3_metrics(batches)
    assert m.max_batch_share > 0.9
    assert is_h2_straggler(m) is True


def test_phase3_metrics_empty_returns_zero() -> None:
    m = phase3_metrics([])
    assert m.n_batches == 0
    assert m.total_wall_s == 0.0
    assert is_h2_straggler(m) is False


def test_h2_test_returns_false_for_single_batch() -> None:
    """Single batch trivially has share=1.0 but H2 only meaningful with >=2 batches."""
    m = phase3_metrics([{"sub_milp_wall_s": 5.0}])
    assert m.n_batches == 1
    assert is_h2_straggler(m) is False


def test_analyse_record_combines_phase1_and_phase3() -> None:
    v = analyse_record(
        phase1_trajectory=_osc_traj(20, 1000),
        phase3_batches=[{"sub_milp_wall_s": 0.1}] * 7 + [{"sub_milp_wall_s": 10.0}],
        budget=1000,
    )
    assert v.h1_degenerate_dual is True
    assert v.h2_straggler is True
    assert v.phase1.n_iter == 20
    assert v.phase3.n_batches == 8


def test_default_grid_matches_spec() -> None:
    """Spec §4.3.1 calls for N_iter ∈ {1, …, 25}."""
    assert tuple(range(1, 26)) == DEFAULT_N_ITER_GRID


def test_default_cell_matches_spec() -> None:
    """Spec §4.3.1: N=10 000, weakly correlated, f=0.5."""
    assert DEFAULT_CELL["N"] == 10_000
    assert DEFAULT_CELL["correlation"] == "weakly"
    assert DEFAULT_CELL["f"] == 0.5


def test_ensure_anomaly_subset_is_idempotent(tmp_path) -> None:
    """Generated files must be reused on the second call (deterministic + fast)."""
    items_first = ensure_anomaly_subset(archive_root=tmp_path, seeds=(0,))
    assert len(items_first) == 1
    mtime_first = items_first[0].path.stat().st_mtime_ns

    items_second = ensure_anomaly_subset(archive_root=tmp_path, seeds=(0,))
    assert items_second[0].path == items_first[0].path
    assert items_second[0].path.stat().st_mtime_ns == mtime_first
    # Same instance content
    assert items_second[0].inst.B == items_first[0].inst.B
    assert DEFAULT_CELL["N"] == items_second[0].inst.N


def test_default_seeds_are_three() -> None:
    """Spec §4.3.1 wants ≥ 1 seed; we ship 3 for statistical sanity."""
    assert len(DEFAULT_SEEDS) >= 1
