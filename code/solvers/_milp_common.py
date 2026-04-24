"""Shared helpers for MILP-based MCKP adapters (HiGHS, SCIP, CBC).

All three solvers model Selective-MCKP identically:

    maximise   sum_{i,j} p_{ij} * x_{ij}
    subject to sum_j x_{ij}            <= 1     for all i  (one-or-zero per class)
               sum_{i,j} c_{ij} * x_{ij} <= B            (budget)
               x_{ij} ∈ {0, 1}

Differences across adapters are confined to the API for adding variables,
constraints, and reading values. This module owns the pieces that are
genuinely shared: indexing, selection extraction, and selection-budget /
profit recomputation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from instances.schema import InstanceModel

from solvers.base import SolveResult, SolverStatus


def iter_pairs(instance: InstanceModel):
    """Yield `(i, j, profit, cost)` for every item in the instance."""
    for i in range(instance.N):
        for j in range(instance.M):
            p, c = instance.items[i][j]
            yield i, j, int(p), int(c)


def extract_selection(
    instance: InstanceModel,
    value_of: Callable[[int, int], float],
    *,
    tol: float = 1e-6,
) -> tuple[dict[int, int | None], int, int]:
    """Round LP/MIP variable values into a feasible Selective-MCKP selection.

    Returns `(items_selected, profit, cost)`. If the solver returns a
    fractional value (rare for pure-binary MIP, but possible after a
    timeout with only a relaxation available), this function still picks
    the largest x_{ij} per class iff that x value exceeds `tol`.
    """
    items_selected: dict[int, int | None] = {}
    profit = 0
    cost = 0
    for i in range(instance.N):
        best_j: int | None = None
        best_val = tol
        for j in range(instance.M):
            v = value_of(i, j)
            if v > best_val:
                best_val = v
                best_j = j
        if best_j is None:
            items_selected[i] = None
            continue
        p, c = instance.items[i][best_j]
        profit += int(p)
        cost += int(c)
        items_selected[i] = best_j
    return items_selected, profit, cost


def make_result(
    *,
    profit: int,
    items_selected: dict[int, int | None],
    total_cost: int,
    wall_time_s: float,
    status: SolverStatus,
    metadata: dict[str, Any],
) -> SolveResult:
    """Convenience wrapper to build a `SolveResult`."""
    return SolveResult(
        profit=profit,
        items_selected=items_selected,
        total_cost=total_cost,
        wall_time_s=wall_time_s,
        status=status,
        solver_metadata=metadata,
    )
