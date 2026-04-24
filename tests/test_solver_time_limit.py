"""Phase C §3.1.3 + §3.2: each solver MUST honour `time_limit_s`.

We solve a deliberately-hard instance with a tiny wall-time budget; every
adapter must return *some* `SolveResult` (not raise, not hang) within a
small multiple of the requested limit.
"""

from __future__ import annotations

import pytest
from solvers import get_solver

from instances import generate_instance

SOLVERS = ["highs", "scip", "cbc"]
TIME_LIMIT = 0.05  # 50 ms — short enough to interrupt CBC's branch-and-bound
GRACE_FACTOR = 30.0  # CBC + PuLP have ~1 s subprocess startup; we accept ≤ 1.5s wall


@pytest.mark.parametrize("name", SOLVERS)
def test_solver_returns_within_grace_period(name: str) -> None:
    """Solver must return a SolveResult, not raise, when starved for time."""
    inst = generate_instance(
        N=400, M=10, correlation="inversely_strongly", f=0.5, seed=0
    )
    solver = get_solver(name)
    result = solver.solve(inst, time_limit_s=TIME_LIMIT, random_seed=0)

    assert result is not None
    grace = max(GRACE_FACTOR * TIME_LIMIT, 1.5)
    assert result.wall_time_s <= grace, (
        f"{name} ignored time_limit_s={TIME_LIMIT}: wall={result.wall_time_s:.3f}s "
        f"(grace cap {grace:.2f}s)"
    )
