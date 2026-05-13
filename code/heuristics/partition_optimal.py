"""Partition-Optimal heuristic for Selective-MCKP.

Algorithm
---------
Split the `N` classes sequentially into `K` (almost) equal-sized
batches. Allocate `B / K` to every batch (rounding down so the total
allocation never exceeds `B`). Solve every batch independently with an
exact MCKP solver (default: HiGHS). Concatenate the per-batch
selections.

This is what §2.2 of the manuscript calls "naive partitioning"; it is
the principal decomposition baseline because it isolates the value of
HLD's data-driven budget allocation: HLD = Partition-Optimal +
λ-driven allocation.

Implementation notes
--------------------
- The default `K` matches the manuscript's running parameterisation
  (`K=8`); callers may override via the `k` constructor argument.
- The per-batch sub-instances are real `InstanceModel`s, so they can
  use any registered exact solver. We default to `"highs"` to keep the
  baseline consistent with the rest of Phase C.
- Class indices are remapped from local (sub-instance) to global before
  building the final `SolveResult`.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from instances.schema import GENERATOR_VERSION, CorrelationKind, InstanceModel
from solvers.base import SolveResult, SolverStatus
from solvers.registry import get_solver, register


class PartitionOptimalAdapter:
    """Equal-split partition-then-exact heuristic."""

    name: str = "partition_optimal"

    def __init__(self, *, k: int = 8, sub_solver: str = "highs") -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self.sub_solver = str(sub_solver)

    def solve(
        self,
        instance: InstanceModel,
        *,
        time_limit_s: float | None = None,
        random_seed: int | None = None,
    ) -> SolveResult:
        t0 = time.perf_counter()

        k = min(self.k, instance.N)
        batch_class_lists = _split_classes(instance.N, k)
        per_batch_budget = instance.B // k

        sub_solver = get_solver(self.sub_solver)

        selected: dict[int, int | None] = {i: None for i in range(instance.N)}
        profit = 0
        cost = 0
        sub_meta: list[dict[str, Any]] = []

        for batch_idx, class_indices in enumerate(batch_class_lists):
            sub_instance = _sub_instance(instance, class_indices, per_batch_budget)
            sub_t0 = time.perf_counter()
            remaining = _remaining_time(time_limit_s, t0)
            sub_result = sub_solver.solve(
                sub_instance,
                time_limit_s=remaining,
                random_seed=random_seed,
            )
            sub_wall = time.perf_counter() - sub_t0

            for local_idx, item_j in sub_result.items_selected.items():
                global_idx = class_indices[local_idx]
                selected[global_idx] = item_j

            profit += sub_result.profit
            cost += sub_result.total_cost
            sub_meta.append(
                {
                    "batch": batch_idx,
                    "n_classes": len(class_indices),
                    "B_k": per_batch_budget,
                    "profit": sub_result.profit,
                    "cost": sub_result.total_cost,
                    "status": str(sub_result.status),
                    "sub_milp_wall_s": sub_wall,
                }
            )

        wall = time.perf_counter() - t0
        meta: dict[str, Any] = {
            "k": k,
            "sub_solver": self.sub_solver,
            "per_batch_budget": per_batch_budget,
            "batches": sub_meta,
            "sub_status_counts": dict(Counter(batch["status"] for batch in sub_meta)),
        }
        status = _aggregate_status(sub_meta)
        return SolveResult(
            profit=profit,
            items_selected=selected,
            total_cost=cost,
            wall_time_s=wall,
            status=status,
            solver_metadata=meta,
        )


def _split_classes(n: int, k: int) -> list[list[int]]:
    """Split `range(n)` into `k` near-equal contiguous batches."""
    base, extra = divmod(n, k)
    out: list[list[int]] = []
    start = 0
    for b in range(k):
        size = base + (1 if b < extra else 0)
        out.append(list(range(start, start + size)))
        start += size
    return out


def _sub_instance(instance: InstanceModel, class_indices: list[int], budget: int) -> InstanceModel:
    """Build an `InstanceModel` covering only `class_indices` with the given budget."""
    items = [list(instance.items[i]) for i in class_indices]
    safe_budget = max(1, int(budget))
    return InstanceModel(
        N=len(class_indices),
        M=instance.M,
        correlation=CorrelationKind(instance.correlation),
        f=instance.f,
        seed=instance.seed,
        B=safe_budget,
        items=items,
        generator_version=GENERATOR_VERSION,
    )


def _remaining_time(limit: float | None, t0: float) -> float | None:
    if limit is None:
        return None
    elapsed = time.perf_counter() - t0
    return max(0.001, float(limit) - elapsed)


def _aggregate_status(sub_meta: list[dict[str, Any]]) -> SolverStatus:
    statuses = {str(batch["status"]) for batch in sub_meta}
    if str(SolverStatus.ERROR) in statuses:
        return SolverStatus.ERROR
    if str(SolverStatus.TIMEOUT) in statuses:
        return SolverStatus.TIMEOUT
    return SolverStatus.FEASIBLE


@register("partition_optimal")
def _factory() -> PartitionOptimalAdapter:
    return PartitionOptimalAdapter()
