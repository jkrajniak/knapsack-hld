"""Tsesmetzis-Roussaki-Sykas (2008) Selective-MCKP heuristic.

Reference
---------
Tsesmetzis, D. T., Roussaki, I., & Sykas, E. D. (2008).
"QoS-aware service evaluation and selection."
European Journal of Operational Research, 191(3), 1101-1112.
DOI: 10.1016/j.ejor.2007.07.015.

Algorithm (two-phase greedy)
----------------------------
The original paper introduces the Selective-MCKP variant in the context
of QoS-aware web-service selection and proposes a two-phase greedy
heuristic. The closed-access full text was not available at
implementation time; the algorithm is reconstructed from the canonical
secondary description in:

    Szkaliczki, T. (2025). "Solution Methods for the Multiple-Choice
    Knapsack Problem and Their Applications."
    Mathematics 13(7), 1097, doi:10.3390/math13071097, §4.5.

Quoting that survey verbatim:

    "Tsesmetzis et al. propose an algorithm for SMCKP based on
    algorithms that solve the Knapsack Problem. First, classes are
    greedily selected with the best value and weight ratios. Then, the
    items of the remaining classes are examined in decreasing order of
    their values, and an item is selected if the capacity constraint is
    not violated."

That description is unambiguous enough to reproduce as:

**Phase A — class-level greedy by best per-class ratio.**
For each class `i`, compute its "champion" item — the item with the
largest profit/cost ratio (ties broken by larger profit). Sort classes
by their champion's ratio in decreasing order, then walk the list and
place each champion item in the knapsack if and only if the remaining
capacity allows. Classes whose champion does not fit are deferred to
Phase B.

**Phase B — item-level greedy by profit on remaining classes.**
Collect every item of every Phase-A-deferred class, sort by profit
descending, and walk the list. An item is selected iff (a) its class
has no Phase-A item, (b) no Phase-B item from the same class has been
selected yet (Selective-MCKP at-most-one constraint), and (c) it fits
in the remaining capacity.

Selective-MCKP semantics
------------------------
A class may end up unselected (item index `None`) if neither phase finds
a fitting item. This is the natural Selective-MCKP behaviour and matches
the SMCKP definition adopted by the original paper.

Caveats
-------
The full-text §5 numerical example from the original paper has not been
transcribed into a unit test (paper unavailable). Validation in
`tests/test_trs2008.py` instead pins (a) the per-phase structural
behaviour on hand-crafted instances and (b) feasibility against the
unified `validate_solution` checker on randomly generated instances.
The §3.4.2 task is left as PENDING in `tasks.md` until the §5 example
can be transcribed from the EJOR paper.
"""

from __future__ import annotations

import math
import time
from typing import Any

from instances.schema import InstanceModel
from solvers.base import SolveResult, SolverStatus
from solvers.registry import register


def _champion_item(class_items: list[list[int]]) -> tuple[int, float] | None:
    """Return `(item_idx, ratio)` of the highest-ratio item in a class.

    Items with `p == 0` are excluded (they cannot improve the objective
    and the original paper's greedy selects only positive-profit items).
    Items with `c == 0 and p > 0` get an infinite ratio so they are
    selected first (consistent with `greedy_max_ratio` and standard
    knapsack heuristics).

    Returns `None` if no positive-profit item exists in the class.
    """
    best_idx: int | None = None
    best_ratio = -math.inf
    best_profit = -1
    for j, (p, c) in enumerate(class_items):
        p, c = int(p), int(c)
        if p <= 0:
            continue
        ratio = math.inf if c == 0 else p / c
        if ratio > best_ratio or (ratio == best_ratio and p > best_profit):
            best_ratio = ratio
            best_profit = p
            best_idx = j
    if best_idx is None:
        return None
    return best_idx, best_ratio


def _phase_a_class_greedy(
    instance: InstanceModel,
    selected: dict[int, int | None],
    remaining_capacity: int,
) -> tuple[int, int, list[int]]:
    """Greedily place each class's champion item in best-ratio order.

    Mutates `selected` in place. Returns `(profit, cost, deferred)`
    where `deferred` is the list of class indices whose champion did
    not fit (and which proceed to Phase B).
    """
    champions: list[tuple[float, int, int, int, int]] = []
    no_champion: list[int] = []
    for i in range(instance.N):
        champ = _champion_item(instance.items[i])
        if champ is None:
            no_champion.append(i)
            continue
        j, ratio = champ
        p, c = instance.items[i][j]
        sort_ratio = float("inf") if math.isinf(ratio) else ratio
        champions.append((sort_ratio, -int(p), i, j, int(c)))

    # Decreasing by ratio; ties broken by larger profit (smaller -p).
    champions.sort(key=lambda t: (-t[0], t[1]))

    profit = 0
    cost = 0
    deferred: list[int] = list(no_champion)
    for _ratio, neg_p, i, j, c in champions:
        if c <= remaining_capacity:
            selected[i] = j
            profit += -neg_p
            cost += c
            remaining_capacity -= c
        else:
            deferred.append(i)
    return profit, cost, deferred


def _phase_b_item_greedy(
    instance: InstanceModel,
    deferred: list[int],
    selected: dict[int, int | None],
    remaining_capacity: int,
) -> tuple[int, int]:
    """Walk items of deferred classes by profit descending, place if it fits.

    Mutates `selected` in place. Returns `(extra_profit, extra_cost)`.
    """
    candidates: list[tuple[int, int, int, int]] = []  # (-profit, cost, cls, item)
    for i in deferred:
        for j, (p, c) in enumerate(instance.items[i]):
            p, c = int(p), int(c)
            if p <= 0:
                continue
            candidates.append((-p, c, i, j))

    # Decreasing profit (ties broken by smaller cost).
    candidates.sort(key=lambda t: (t[0], t[1]))

    extra_profit = 0
    extra_cost = 0
    for neg_p, c, i, j in candidates:
        if selected[i] is not None:
            continue
        if c > remaining_capacity:
            continue
        selected[i] = j
        extra_profit += -neg_p
        extra_cost += c
        remaining_capacity -= c
    return extra_profit, extra_cost


class Trs2008Adapter:
    """Pure-Python re-implementation of the TRS-2008 SMCKP heuristic."""

    name: str = "trs2008"

    def solve(
        self,
        instance: InstanceModel,
        *,
        time_limit_s: float | None = None,
        random_seed: int | None = None,
    ) -> SolveResult:
        del time_limit_s, random_seed  # heuristic is deterministic and O(NM log NM)
        t0 = time.perf_counter()

        selected: dict[int, int | None] = {i: None for i in range(instance.N)}
        budget = int(instance.B)

        a_profit, a_cost, deferred = _phase_a_class_greedy(instance, selected, budget)
        b_profit, b_cost = _phase_b_item_greedy(instance, deferred, selected, budget - a_cost)

        total_profit = a_profit + b_profit
        total_cost = a_cost + b_cost
        wall = time.perf_counter() - t0

        meta: dict[str, Any] = {
            "phase_a_profit": a_profit,
            "phase_a_cost": a_cost,
            "phase_a_classes_selected": sum(
                1 for i in range(instance.N) if selected[i] is not None and i not in deferred
            ),
            "phase_b_profit": b_profit,
            "phase_b_cost": b_cost,
            "phase_b_classes_deferred": len(deferred),
            "algorithm": "trs2008-two-phase-greedy",
        }

        return SolveResult(
            profit=total_profit,
            items_selected=selected,
            total_cost=total_cost,
            wall_time_s=wall,
            status=SolverStatus.FEASIBLE,
            solver_metadata=meta,
        )


@register("trs2008")
def _factory() -> Trs2008Adapter:
    return Trs2008Adapter()
