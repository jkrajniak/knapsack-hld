"""Solver protocol and the canonical `SolveResult` dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from instances.schema import InstanceModel


class SolverStatus(StrEnum):
    """Outcome of a single `Solver.solve` call."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class SolveResult:
    """Canonical output of every solver and heuristic.

    `items_selected` is a mapping `class_idx -> item_idx | None`.
    Selective-MCKP allows leaving a class unselected, which is encoded
    as `None`. Classic MCKP solvers MUST select exactly one item per
    class (no `None` values).
    """

    profit: int
    items_selected: dict[int, int | None]
    total_cost: int
    wall_time_s: float
    status: SolverStatus
    solver_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_classes_selected(self) -> int:
        return sum(1 for v in self.items_selected.values() if v is not None)


@runtime_checkable
class Solver(Protocol):
    """Common interface for every exact solver, heuristic, and HLD."""

    name: str

    def solve(
        self,
        instance: InstanceModel,
        *,
        time_limit_s: float | None = None,
        random_seed: int | None = None,
    ) -> SolveResult:  # pragma: no cover -- protocol body
        ...


class InvalidSolutionError(ValueError):
    """Raised by `validate_solution` when a result violates the problem."""


def validate_solution(instance: InstanceModel, result: SolveResult) -> None:
    """Re-verify a `SolveResult` against the instance.

    Checks performed:
    1. Every class index is in `[0, N)` and item indices are in `[0, M)`.
    2. Each class is selected at most once (one item or `None`).
    3. The recomputed total cost matches `result.total_cost` and is `≤ B`.
    4. The recomputed total profit matches `result.profit`.
    """
    profit = 0
    cost = 0
    for cls_idx, item_idx in result.items_selected.items():
        if not 0 <= cls_idx < instance.N:
            raise InvalidSolutionError(f"class index {cls_idx} out of range [0, {instance.N})")
        if item_idx is None:
            continue
        if not 0 <= item_idx < instance.M:
            raise InvalidSolutionError(
                f"item index {item_idx} out of range [0, {instance.M}) in class {cls_idx}"
            )
        p, c = instance.items[cls_idx][item_idx]
        profit += p
        cost += c

    if cost != result.total_cost:
        raise InvalidSolutionError(f"recomputed cost {cost} != reported {result.total_cost}")
    if profit != result.profit:
        raise InvalidSolutionError(f"recomputed profit {profit} != reported {result.profit}")
    if cost > instance.B:
        raise InvalidSolutionError(f"total cost {cost} exceeds budget {instance.B}")
