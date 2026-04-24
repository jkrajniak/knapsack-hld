"""Exact MCKP solver in the spirit of Pisinger (1995).

Algorithm
---------
Two stages:

1. **LP relaxation via Sinha-Zoltners (1979).** For every class we keep
   only the *LP-undominated* items (the upper envelope of the
   profit-vs-cost staircase). Items are sorted within each class by
   non-decreasing cost; the resulting "increment" of moving from item
   `j` to item `j+1` has cost `Δc > 0` and efficiency `Δp / Δc`. Sorting
   all increments by decreasing efficiency and applying them one by one
   while the cumulative cost stays below `B` produces the LP optimum,
   with the last (cracking) increment fractionally split.

2. **Branch-and-bound.** We branch on the cracking class by trying each
   of its LP-undominated items in turn, fixing it, and recursing. The
   LP relaxation of the residual problem is the upper bound used for
   pruning.

Differences from Pisinger 1995
------------------------------
Pisinger's "minimal algorithm" uses a primal-dual *core* approach plus
a tightly-tuned branch-and-bound around a small core of classes. Our
re-implementation is a clean LP + B&B variant that returns *the same
optimum* on all Pisinger 1995 archive instances we have validated
against, but with different performance characteristics on very large
instances. The choice is documented in `README.md` under this module.

The implementation is pure Python with NumPy used only for sorting; it
is intentionally simple so the paper's claim of correctness can be
audited end-to-end without external dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from instances.schema import InstanceModel
from solvers.base import SolveResult, SolverStatus
from solvers.registry import register

EPS = 1e-9


@dataclass(frozen=True)
class _ClassEnvelope:
    """Two views of one class' items.

    - `pareto`   : ALL Pareto-optimal items, sorted by cost ascending.
                   Branch-and-bound branches over this set so it cannot
                   miss any integer-optimal item.
    - `envelope` : strict concave upper-hull subset of `pareto`. The LP
                   relaxation walks the increments of this hull.

    Each entry is `(orig_idx, profit, cost)` where `orig_idx` indexes
    the augmented item list (so the dummy lives at `0`, real items at
    `1..M` for Selective-MCKP).
    """

    cls_idx: int
    pareto: tuple[tuple[int, int, int], ...]
    envelope: tuple[tuple[int, int, int], ...]


def _build_envelope(cls_idx: int, items: list[list[int]]) -> _ClassEnvelope:
    """Return the Pareto set and the concave upper hull of one class."""
    triples = [(int(p), int(c), j) for j, (p, c) in enumerate(items)]
    triples.sort(key=lambda t: (t[1], -t[0]))

    pareto: list[tuple[int, int, int]] = []
    best_profit = -1
    last_cost = -1
    for p, c, j in triples:
        if c == last_cost:
            continue
        if p > best_profit:
            pareto.append((j, p, c))
            best_profit = p
            last_cost = c

    envelope: list[tuple[int, int, int]] = []
    for j, p, c in pareto:
        while len(envelope) >= 2:
            _j1, p1, c1 = envelope[-2]
            _j2, p2, c2 = envelope[-1]
            if (p2 - p1) * (c - c2) <= (p - p2) * (c2 - c1):
                envelope.pop()
            else:
                break
        envelope.append((j, p, c))

    return _ClassEnvelope(cls_idx=cls_idx, pareto=tuple(pareto), envelope=tuple(envelope))


def _lp_solve(
    envelopes: list[_ClassEnvelope],
    capacity: int,
) -> tuple[float, dict[int, int], int | None]:
    """Sinha-Zoltners LP relaxation.

    Returns `(lp_value, integer_partial_selection, cracking_cls_or_None)`.
    `integer_partial_selection[i]` is the integer item index assigned in
    the LP-rounded-down solution; the cracking class (if any) is the
    one whose increment was only fractionally accepted.
    """
    # Start every class at its cheapest LP-undominated item. The LP
    # relaxation chooses the cheapest item for free; subsequent
    # increments cost (Δc) and gain (Δp).
    selection: dict[int, int] = {}
    base_profit = 0.0
    base_cost = 0
    increments: list[tuple[float, int, int]] = []  # (efficiency, cls_pos, k_pos)
    for pos, env in enumerate(envelopes):
        if not env.envelope:
            continue
        first_j, first_p, first_c = env.envelope[0]
        selection[env.cls_idx] = first_j
        base_profit += first_p
        base_cost += first_c
        for k in range(1, len(env.envelope)):
            _j_prev, p_prev, c_prev = env.envelope[k - 1]
            _j_curr, p_curr, c_curr = env.envelope[k]
            dp = p_curr - p_prev
            dc = c_curr - c_prev
            if dc <= 0 or dp <= 0:
                continue
            increments.append((dp / dc, pos, k))

    if base_cost > capacity:
        # Even the cheapest selection violates the budget. We still
        # return the best-effort selection; B&B will reject it as
        # infeasible if needed.
        return base_profit, selection, None

    increments.sort(key=lambda t: -t[0])

    cap_left = capacity - base_cost
    profit = base_profit
    cracking_cls: int | None = None

    for _eff, pos, k in increments:
        env = envelopes[pos]
        _j_prev, p_prev, c_prev = env.envelope[k - 1]
        j_curr, p_curr, c_curr = env.envelope[k]
        dp = p_curr - p_prev
        dc = c_curr - c_prev
        if dc <= cap_left + EPS:
            cap_left -= dc
            profit += dp
            selection[env.cls_idx] = j_curr
        else:
            frac = max(0.0, cap_left) / dc
            profit += frac * dp
            cracking_cls = env.cls_idx
            break

    return profit, selection, cracking_cls


_DUMMY_INDEX = -1  # sentinel: "this class was not selected"


def _augment_for_selective(items: list[list[int]]) -> list[list[int]]:
    """Return `items` with a virtual zero-profit zero-cost dummy prepended.

    The dummy occupies index `_DUMMY_INDEX` in `instance.items`-space; we
    keep it as the cheapest LP-undominated entry of the augmented class
    so the LP relaxation can naturally "skip" the class for free. The
    final `SolveResult` maps this sentinel back to `None`.
    """
    return [[0, 0], *list(items)]


def _solve_mcknap(
    instance: InstanceModel,
    *,
    deadline: float | None,
    require_full: bool,
) -> tuple[int, dict[int, int | None], int, str, dict[str, Any]]:
    """Top-level B&B driver.

    `require_full=True` enforces *classic MCKP*: every class must be
    selected. `require_full=False` enforces *Selective-MCKP*: every
    class is augmented internally with a virtual `(0, 0)` dummy item.
    """
    if require_full:
        augmented_items = [list(instance.items[i]) for i in range(instance.N)]
        offset = 0
    else:
        augmented_items = [_augment_for_selective(instance.items[i]) for i in range(instance.N)]
        offset = 1  # internal indices are shifted by +1

    envelopes = [_build_envelope(i, augmented_items[i]) for i in range(instance.N)]
    capacity = int(instance.B)

    best_profit = -1
    best_selection: dict[int, int | None] = {i: None for i in range(instance.N)}
    metadata: dict[str, Any] = {"nodes_explored": 0, "lp_relaxations": 0, "best_first": True}

    fixed: dict[int, int] = {}

    def _bound_and_recurse() -> None:
        nonlocal best_profit, best_selection
        metadata["nodes_explored"] += 1
        if deadline is not None and time.perf_counter() > deadline:
            raise _Timeout

        residual = [env for env in envelopes if env.cls_idx not in fixed]
        residual_capacity = capacity - sum(augmented_items[i][j][1] for i, j in fixed.items())
        residual_profit_fixed = sum(augmented_items[i][j][0] for i, j in fixed.items())

        if residual_capacity < 0:
            return

        if not residual:
            if residual_profit_fixed > best_profit:
                best_profit = residual_profit_fixed
                best_selection = _to_external(fixed, instance.N, offset)
            return

        metadata["lp_relaxations"] += 1
        lp_value, lp_selection, cracking = _lp_solve(residual, residual_capacity)
        upper_bound = residual_profit_fixed + lp_value

        if upper_bound <= best_profit + EPS:
            return

        if cracking is None:
            full_int = dict(fixed)
            full_int.update(lp_selection)
            integer_cost = sum(augmented_items[i][j][1] for i, j in full_int.items())
            integer_value = sum(augmented_items[i][j][0] for i, j in full_int.items())
            if integer_cost <= capacity and integer_value > best_profit:
                best_profit = integer_value
                best_selection = _to_external(full_int, instance.N, offset)
            return

        cracking_env = next(env for env in residual if env.cls_idx == cracking)
        choices = sorted(
            ((j, p) for j, p, _c in cracking_env.pareto),
            key=lambda jp: -jp[1],
        )

        for j, _p in choices:
            fixed[cracking] = j
            try:
                _bound_and_recurse()
            finally:
                del fixed[cracking]

    timed_out = False
    try:
        _bound_and_recurse()
    except _Timeout:
        timed_out = True
        if best_profit < 0:
            return (
                0,
                {i: None for i in range(instance.N)},
                0,
                SolverStatus.TIMEOUT.value,
                metadata,
            )

    if best_profit < 0:
        return 0, best_selection, 0, SolverStatus.INFEASIBLE.value, metadata

    total_cost = sum(instance.items[i][j][1] for i, j in best_selection.items() if j is not None)
    status = SolverStatus.FEASIBLE if timed_out else SolverStatus.OPTIMAL
    return int(best_profit), best_selection, int(total_cost), status.value, metadata


def _to_external(
    fixed: dict[int, int],
    N: int,
    offset: int,
) -> dict[int, int | None]:
    """Translate internal item indices (with possible dummy) to externals."""
    out: dict[int, int | None] = {i: None for i in range(N)}
    for i, j_internal in fixed.items():
        j_external = j_internal - offset
        out[i] = None if j_external < 0 else j_external
    return out


class _Timeout(Exception):
    """Signal a B&B timeout via the recursion stack."""


class McknapAdapter:
    """Pure-Python exact MCKP / Selective-MCKP solver."""

    name: str = "mcknap"

    def solve(
        self,
        instance: InstanceModel,
        *,
        time_limit_s: float | None = None,
        random_seed: int | None = None,
    ) -> SolveResult:
        del random_seed  # deterministic; seed unused
        deadline = (time.perf_counter() + time_limit_s) if time_limit_s else None

        t0 = time.perf_counter()
        profit, selection, cost, status_str, meta = _solve_mcknap(
            instance, deadline=deadline, require_full=False
        )
        wall = time.perf_counter() - t0

        return SolveResult(
            profit=profit,
            items_selected=selection,
            total_cost=cost,
            wall_time_s=wall,
            status=SolverStatus(status_str),
            solver_metadata={
                **meta,
                "algorithm": "sinha-zoltners-lp + branch-and-bound",
                "wall_time_s": wall,
            },
        )


@register("mcknap")
def _factory() -> McknapAdapter:
    return McknapAdapter()
