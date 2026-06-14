"""Hybrid Lagrangian-Decomposition (HLD) solver for Selective-MCKP.

Algorithm
---------
HLD has three phases (manuscript §2.4-§2.7).

**Phase 1 — Global shadow price estimation.** Binary search for the
Lagrange multiplier `lambda_est` over `[0, lambda_max]` for `N_iter`
iterations. At each iteration we solve the per-class Lagrangian
`max_j (p_ij - lambda * c_ij)` (Selective-MCKP picks `j` only if the
value is positive), accumulate the total cost, and bisect:
`cost > B  =>  lambda_lo = lambda_mid`,
`cost <= B =>  lambda_hi = lambda_mid`.

**Phase 2 — Data-driven budget allocation.** Partition the N classes
sequentially into K equal-sized batches. For each batch, compute the
Phase-1 selection-based estimated cost `C_k`. The per-batch budget is

    B_k = B * (1 - alpha) / K                 (equal-allocation share)
        + B * alpha * C_k / C_total           (proportional share)

with a fallback to `B / K` if `C_total == 0`. The mixed allocation
captures the information from `lambda_est` while staying robust to
estimation error (alpha = 0.9 is the manuscript default).

**Phase 3 — Parallel optimal solving.** Each batch is its own
Selective-MCKP sub-instance solved by an exact MILP (HiGHS by default).
Per-batch wall time, batch size, and budget are logged.

Instance-dependent lambda_max
-----------------------------
A fixed constant `lambda_max = 10` is opaque and instance-insensitive.
Instead we set

    lambda_max = ceil(max_{i, j} p_ij / c_ij)

with `c_ij == 0` items excluded (they correspond to free profit and
do not constrain lambda from above). This makes the binary-search range
instance-aware and removes the magic constant from §2.7.

Solver-metadata schema
----------------------
- `lambda_est`           — final lambda from Phase 1
- `lambda_max`           — instance-dependent upper bound used
- `phase1_trajectory`    — list of dicts with keys
    `(iter, lambda_lo, lambda_mid, lambda_hi, total_cost)`
- `phase2_allocation`    — list of dicts with keys
    `(batch, B_k, estimated_cost, n_classes)`
- `phase3_batches`       — list of dicts with keys
    `(batch, B_k, n_items, sub_milp_wall_s, profit, cost, status)`
- `fallback_equal_split` — bool, true iff Phase 2 hit the C_total == 0 fallback
- `params`               — dict of (`N_iter`, `alpha`, `K`, `lambda_max`, `sub_solver`)
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Literal

from instances.schema import GENERATOR_VERSION, CorrelationKind, InstanceModel

from solvers.base import SolveResult, SolverStatus
from solvers.registry import get_solver, register

DEFAULT_N_ITER = 20
DEFAULT_ALPHA = 0.9
DEFAULT_K = 8
DEFAULT_SUB_SOLVER = "highs"

ClassOrdering = Literal["sequential", "random", "adversarial"]
CLASS_ORDERINGS: tuple[ClassOrdering, ...] = ("sequential", "random", "adversarial")
DEFAULT_CLASS_ORDERING: ClassOrdering = "sequential"


class HldAdapter:
    """Hybrid Lagrangian-Decomposition with full Phase-1/2/3 instrumentation."""

    name: str = "hld"

    def __init__(
        self,
        *,
        n_iter: int = DEFAULT_N_ITER,
        alpha: float = DEFAULT_ALPHA,
        k: int = DEFAULT_K,
        sub_solver: str = DEFAULT_SUB_SOLVER,
        lambda_max_override: float | None = None,
        class_ordering: ClassOrdering = DEFAULT_CLASS_ORDERING,
    ) -> None:
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if class_ordering not in CLASS_ORDERINGS:
            raise ValueError(
                f"class_ordering must be one of {CLASS_ORDERINGS}, got {class_ordering!r}"
            )
        self.n_iter = int(n_iter)
        self.alpha = float(alpha)
        self.k = int(k)
        self.sub_solver = str(sub_solver)
        self.lambda_max_override = lambda_max_override
        self.class_ordering: ClassOrdering = class_ordering

    def solve(
        self,
        instance: InstanceModel,
        *,
        time_limit_s: float | None = None,
        random_seed: int | None = None,
    ) -> SolveResult:
        t0 = time.perf_counter()
        deadline = None if time_limit_s is None else t0 + float(time_limit_s)

        lambda_max = self._lambda_max(instance)

        lambda_est, phase1_trajectory = _phase1_binary_search(
            instance, lambda_max=lambda_max, n_iter=self.n_iter
        )

        k = min(self.k, instance.N)
        class_order = _class_order(instance, ordering=self.class_ordering, random_seed=random_seed)
        batches = _split_classes(instance.N, k, order=class_order)
        per_batch_estimated, fallback = _phase2_estimate(
            instance, batches=batches, lambda_est=lambda_est
        )
        per_batch_budget = _phase2_allocate(
            B=instance.B,
            k=k,
            per_batch_estimated=per_batch_estimated,
            alpha=self.alpha,
            fallback=fallback,
        )
        phase2_log = [
            {
                "batch": b,
                "B_k": per_batch_budget[b],
                "estimated_cost": per_batch_estimated[b],
                "n_classes": len(batches[b]),
            }
            for b in range(k)
        ]

        sub = get_solver(self.sub_solver)
        selected: dict[int, int | None] = {i: None for i in range(instance.N)}
        profit = 0
        cost = 0
        phase3_log: list[dict[str, Any]] = []

        for b, class_indices in enumerate(batches):
            sub_inst = _sub_instance(instance, class_indices, per_batch_budget[b])
            sub_t0 = time.perf_counter()
            remaining = _remaining_time(deadline)
            sub_res = sub.solve(sub_inst, time_limit_s=remaining, random_seed=random_seed)
            sub_wall = time.perf_counter() - sub_t0
            for local_i, item_j in sub_res.items_selected.items():
                selected[class_indices[local_i]] = item_j
            profit += sub_res.profit
            cost += sub_res.total_cost
            phase3_log.append(
                {
                    "batch": b,
                    "B_k": per_batch_budget[b],
                    "n_items": sub_res.n_classes_selected,
                    "sub_milp_wall_s": sub_wall,
                    "profit": sub_res.profit,
                    "cost": sub_res.total_cost,
                    "status": str(sub_res.status),
                }
            )

        wall = time.perf_counter() - t0
        meta: dict[str, Any] = {
            "lambda_est": lambda_est,
            "lambda_max": lambda_max,
            "phase1_trajectory": phase1_trajectory,
            "phase2_allocation": phase2_log,
            "phase3_batches": phase3_log,
            "fallback_equal_split": fallback,
            "params": {
                "n_iter": self.n_iter,
                "alpha": self.alpha,
                "k": k,
                "lambda_max": lambda_max,
                "sub_solver": self.sub_solver,
                "class_ordering": self.class_ordering,
            },
        }
        status = (
            SolverStatus.TIMEOUT
            if deadline is not None and time.perf_counter() > deadline
            else SolverStatus.FEASIBLE
        )
        return SolveResult(
            profit=profit,
            items_selected=selected,
            total_cost=cost,
            wall_time_s=wall,
            status=status,
            solver_metadata=meta,
        )

    def _lambda_max(self, instance: InstanceModel) -> float:
        if self.lambda_max_override is not None:
            return float(self.lambda_max_override)
        return _instance_dependent_lambda_max(instance)


def _instance_dependent_lambda_max(instance: InstanceModel) -> float:
    """`lambda_max = ceil(max_{i, j} p_ij / c_ij)` with `c_ij == 0` skipped.

    A class with only c==0 items is degenerate (no upper-bound contribution
    from cost); we floor `lambda_max` at 1.0 in that case so the binary
    search still has a non-empty range.
    """
    best_ratio = 0.0
    for i in range(instance.N):
        for j in range(instance.M):
            p, c = instance.items[i][j]
            if c <= 0:
                continue
            ratio = float(p) / float(c)
            if ratio > best_ratio:
                best_ratio = ratio
    if best_ratio <= 0.0:
        return 1.0
    return float(math.ceil(best_ratio))


def _phase1_binary_search(
    instance: InstanceModel, *, lambda_max: float, n_iter: int
) -> tuple[float, list[dict[str, float | int]]]:
    """Binary-search lambda over `[0, lambda_max]` with full per-iteration logging."""
    lo = 0.0
    hi = float(lambda_max)
    trajectory: list[dict[str, float | int]] = []
    final_mid = hi
    for it in range(n_iter):
        mid = (lo + hi) / 2.0
        total_cost = _selective_lagrangian_cost(instance, mid)
        trajectory.append(
            {
                "iter": it,
                "lambda_lo": lo,
                "lambda_mid": mid,
                "lambda_hi": hi,
                "total_cost": total_cost,
            }
        )
        if total_cost > instance.B:
            lo = mid
        else:
            hi = mid
        final_mid = mid
    lambda_est = hi if hi > 0 else final_mid
    return lambda_est, trajectory


def _selective_lagrangian_cost(instance: InstanceModel, lam: float) -> int:
    """Selective-MCKP Lagrangian: pick `j*(i)` only if `p - lam*c > 0`.

    Returns the sum of `c_ij*` over the picks.
    """
    total = 0
    for i in range(instance.N):
        best_val = 0.0
        best_c = 0
        for j in range(instance.M):
            p, c = instance.items[i][j]
            val = float(p) - lam * float(c)
            if val > best_val:
                best_val = val
                best_c = int(c)
        total += best_c
    return total


def _phase2_estimate(
    instance: InstanceModel, *, batches: list[list[int]], lambda_est: float
) -> tuple[list[int], bool]:
    """Compute per-batch estimated cost from the Phase-1 selection."""
    per_batch: list[int] = []
    for class_indices in batches:
        c_k = 0
        for i in class_indices:
            best_val = 0.0
            best_c = 0
            for j in range(instance.M):
                p, c = instance.items[i][j]
                val = float(p) - lambda_est * float(c)
                if val > best_val:
                    best_val = val
                    best_c = int(c)
            c_k += best_c
        per_batch.append(c_k)
    fallback = sum(per_batch) == 0
    return per_batch, fallback


def _phase2_allocate(
    *,
    B: int,
    k: int,
    per_batch_estimated: list[int],
    alpha: float,
    fallback: bool,
) -> list[int]:
    """Mixed allocation: equal share + proportional share."""
    if fallback:
        equal = B // k
        return [equal] * k
    total_est = sum(per_batch_estimated)
    b_equal = B * (1.0 - alpha) / k
    b_prop = B * alpha
    out: list[int] = []
    for c_k in per_batch_estimated:
        share = b_equal + b_prop * (c_k / total_est)
        out.append(max(1, math.floor(share)))
    return out


def _split_classes(n: int, k: int, *, order: list[int] | None = None) -> list[list[int]]:
    """Split `n` class indices into `k` contiguous batches along `order`.

    Default `order=None` keeps the legacy `range(n)` (sequential) behaviour.
    Pass a permutation of `range(n)` (produced by `_class_order`) to apply
    a non-sequential ordering before the equal-size split.
    """
    indices = list(range(n)) if order is None else list(order)
    if len(indices) != n or sorted(indices) != list(range(n)):
        raise ValueError(f"order must be a permutation of range({n}); got len={len(indices)}")
    base, extra = divmod(n, k)
    out: list[list[int]] = []
    start = 0
    for b in range(k):
        size = base + (1 if b < extra else 0)
        out.append(indices[start : start + size])
        start += size
    return out


def _class_order(
    instance: InstanceModel,
    *,
    ordering: ClassOrdering,
    random_seed: int | None,
) -> list[int]:
    """Return a permutation of `range(N)` per the requested ordering.

    - `sequential` is the identity: `[0, 1, ..., N-1]`.
    - `random` uses an instance-local `random.Random(random_seed)` so the
      shuffle is reproducible from `random_seed` alone (independent of
      any global RNG state). When `random_seed is None` we derive a
      stable seed from `instance.seed` so the same instance still gives
      the same shuffle across runs.
    - `adversarial` sorts classes by descending max-profit-per-cost
      ratio with the class index as a stable tie-breaker. The highest
      profit-per-cost class therefore lands in batch 0, which stresses
      any equal-budget decomposition and exercises HLD's Phase-2
      proportional-allocation path. Classes with no positive-cost item
      receive ratio 0 and sort to the end.
    """
    if ordering == "sequential":
        return list(range(instance.N))
    if ordering == "random":
        seed = random_seed if random_seed is not None else int(instance.seed)
        rng = random.Random(seed)
        order = list(range(instance.N))
        rng.shuffle(order)
        return order
    if ordering == "adversarial":

        def max_pc_ratio(i: int) -> float:
            return max(
                (float(p) / float(c) for (p, c) in instance.items[i] if c > 0),
                default=0.0,
            )

        return sorted(range(instance.N), key=lambda i: (-max_pc_ratio(i), i))
    raise ValueError(f"unknown class_ordering: {ordering!r}")


def _sub_instance(instance: InstanceModel, class_indices: list[int], budget: int) -> InstanceModel:
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


def _remaining_time(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.001, deadline - time.perf_counter())


@register("hld")
def _factory() -> HldAdapter:
    return HldAdapter()
