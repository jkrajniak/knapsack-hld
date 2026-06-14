"""BISSA — Bi-objective Approximate Solution Search Algorithm.

Reference
---------
Bednarczuk, E. M., Miroforidis, J. & Pyzel, P. (2018).
"A multi-criteria approach to approximate solution of multiple-choice
knapsack problem." *Computational Optimization and Applications*, 70(3),
889-910. https://doi.org/10.1007/s10589-018-9988-z

Algorithm summary
-----------------
BISSA bi-objectivises MCKP into

    max  c^T x  (profit)
    max  -w^T x + b  (slack to budget)
    s.t. one-or-zero per class, x in {0, 1}.

It then solves a series of linearly-scalarised problems

    BS(lambda):  max (1 - lambda) * c^T x - lambda * w^T x

for `lambda` in `(0, 1)`. Crucially, BS(lambda) decomposes by class and
admits a closed-form per-class solution: pick the item that maximises
`(1 - lambda) * p_ij - lambda * c_ij`. The algorithm bisects `lambda`
to find the smallest value at which BS(lambda)'s solution is feasible
for the original budget. That solution is the BISSA approximation.

Selective-MCKP transformation
-----------------------------
The original BISSA paper formulates classic MCKP, which forces *exactly
one* item per class. To use BISSA on Selective-MCKP we apply the
standard transformation: each class is augmented with a dummy item
`(p=0, c=0)`. Picking the dummy is BISSA's encoding of "skip this
class". The transformation is recorded in `solver_metadata` under the
key `"transformation"` so it is auditable from the result alone.
"""

from __future__ import annotations

import time
from typing import Any

from instances.schema import InstanceModel
from solvers.base import SolveResult, SolverStatus
from solvers.registry import register

EPS = 1e-12
DEFAULT_MAX_ITERS = 50


class BissaAdapter:
    """BISSA with the explicit MCKP -> Selective-MCKP dummy-item transformation."""

    name: str = "bissa"

    def __init__(self, *, max_iters: int = DEFAULT_MAX_ITERS) -> None:
        if max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        self.max_iters = int(max_iters)

    def solve(
        self,
        instance: InstanceModel,
        *,
        time_limit_s: float | None = None,
        random_seed: int | None = None,
    ) -> SolveResult:
        del random_seed
        t0 = time.perf_counter()
        deadline = None if time_limit_s is None else t0 + float(time_limit_s)

        augmented = _augment_with_dummy(instance.items)

        best: tuple[int, int, dict[int, int]] | None = None
        lo, hi = 0.0, 1.0
        trajectory: list[dict[str, float | int]] = []

        for it in range(self.max_iters):
            lam = (lo + hi) / 2.0
            sel, profit, cost = _solve_scalarisation(augmented, lam)
            trajectory.append({"iter": it, "lambda": lam, "profit": profit, "cost": cost})
            if cost <= instance.B:
                if best is None or profit > best[0]:
                    best = (profit, cost, sel)
                hi = lam
            else:
                lo = lam
            if hi - lo < 1e-9:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break

        if best is None:
            sel = {i: 0 for i in range(instance.N)}
            best = (0, 0, sel)

        items_selected = _decode_selection(best[2], instance.M)

        wall = time.perf_counter() - t0
        meta: dict[str, Any] = {
            "transformation": "mckp_to_selective_mckp_dummy_item",
            "n_iterations": len(trajectory),
            "trajectory": trajectory,
            "final_lambda_window": [lo, hi],
        }
        status = (
            SolverStatus.FEASIBLE
            if deadline is None or time.perf_counter() < deadline
            else SolverStatus.TIMEOUT
        )
        return SolveResult(
            profit=best[0],
            items_selected=items_selected,
            total_cost=best[1],
            wall_time_s=wall,
            status=status,
            solver_metadata=meta,
        )


def _augment_with_dummy(items: list) -> list[list[tuple[int, int]]]:
    """Prepend a `(0, 0)` dummy item to every class.

    The dummy becomes index `0` in the augmented list; real items
    shift to indices `1..M`. This is the standard transformation that
    lets a classic MCKP solver handle Selective-MCKP.
    """
    out: list[list[tuple[int, int]]] = []
    for cls in items:
        augmented_cls: list[tuple[int, int]] = [(0, 0)]
        augmented_cls.extend((int(p), int(c)) for p, c in cls)
        out.append(augmented_cls)
    return out


def _solve_scalarisation(
    augmented: list[list[tuple[int, int]]], lam: float
) -> tuple[dict[int, int], int, int]:
    """Per-class closed-form maximiser of `(1 - lam) * p - lam * c`.

    Ties broken in favour of higher profit, then lower cost, then the
    dummy (j=0) — this keeps Selective-MCKP behaviour deterministic at
    `lam` boundaries.
    """
    one_minus = 1.0 - lam
    sel: dict[int, int] = {}
    total_p = 0
    total_c = 0
    for i, cls in enumerate(augmented):
        best_score = -float("inf")
        best_p = -1
        best_c = float("inf")
        best_j = 0
        for j, (p, c) in enumerate(cls):
            score = one_minus * p - lam * c
            if score > best_score + EPS or (
                abs(score - best_score) < EPS and (p > best_p or (p == best_p and c < best_c))
            ):
                best_score = score
                best_p = p
                best_c = c
                best_j = j
        sel[i] = best_j
        total_p += best_p
        total_c += best_c
    return sel, total_p, total_c


def _decode_selection(augmented_sel: dict[int, int], real_m: int) -> dict[int, int | None]:
    """Map augmented indices back to original Selective-MCKP indices.

    Augmented index `0` is the dummy `=> None`. Augmented indices
    `1..M` map to real indices `0..M-1`.
    """
    decoded: dict[int, int | None] = {}
    for cls_idx, aug_j in augmented_sel.items():
        if aug_j == 0:
            decoded[cls_idx] = None
        else:
            real_j = aug_j - 1
            if not 0 <= real_j < real_m:
                raise RuntimeError(f"BISSA decoded out-of-range item {real_j} for class {cls_idx}")
            decoded[cls_idx] = real_j
    return decoded


@register("bissa")
def _factory() -> BissaAdapter:
    return BissaAdapter()
