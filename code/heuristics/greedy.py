"""Greedy-MaxRatio heuristic for Selective-MCKP.

Algorithm
---------
1. Enumerate every (class, item) pair as a candidate `(i, j, p, c)`.
2. Sort candidates by profit/cost ratio descending. Items with `c == 0`
   and `p > 0` are placed first (infinite ratio); items with `p == 0`
   are placed last (zero ratio).
3. Walk the sorted list. Accept a candidate iff its class has not been
   selected yet AND adding `c` keeps the running cost within `B`.

This is a single-pass O(N M log(N M)) procedure. It is the standard
weak baseline for MCKP / Selective-MCKP and is documented in §2.2 of
the manuscript and in Akcay et al. 2007.

Selective-MCKP semantics: a class may end up unselected if no item with
positive ratio fits — this is correct behaviour for Selective-MCKP and
matches the manuscript's framing.
"""

from __future__ import annotations

import time
from typing import Any

from instances.schema import InstanceModel
from solvers.base import SolveResult, SolverStatus
from solvers.registry import register


class GreedyMaxRatioAdapter:
    """Single-pass profit/cost-ratio greedy."""

    name: str = "greedy_max_ratio"

    def solve(
        self,
        instance: InstanceModel,
        *,
        time_limit_s: float | None = None,
        random_seed: int | None = None,
    ) -> SolveResult:
        del time_limit_s, random_seed
        t0 = time.perf_counter()

        candidates: list[tuple[float, int, int, int, int]] = []
        for i in range(instance.N):
            for j in range(instance.M):
                p, c = instance.items[i][j]
                p, c = int(p), int(c)
                if p <= 0:
                    continue
                ratio = float("inf") if c == 0 else p / c
                candidates.append((ratio, -p, i, j, c))

        candidates.sort(key=lambda t: (-t[0], t[1]))

        selected: dict[int, int | None] = {i: None for i in range(instance.N)}
        profit = 0
        cost = 0
        for _ratio, neg_p, i, j, c in candidates:
            if selected[i] is not None:
                continue
            if cost + c > instance.B:
                continue
            selected[i] = j
            profit += -neg_p
            cost += c

        wall = time.perf_counter() - t0
        meta: dict[str, Any] = {"n_candidates": len(candidates)}
        return SolveResult(
            profit=profit,
            items_selected=selected,
            total_cost=cost,
            wall_time_s=wall,
            status=SolverStatus.FEASIBLE,
            solver_metadata=meta,
        )


@register("greedy_max_ratio")
def _factory() -> GreedyMaxRatioAdapter:
    return GreedyMaxRatioAdapter()
