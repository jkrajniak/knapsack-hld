"""Unified solver and heuristic interface for Selective-MCKP.

Every exact solver, every heuristic, and HLD itself implements the same
`Solver` protocol and returns a `SolveResult`. This guarantees that
benchmark scripts, the SMAC3 tuning harness, and the paper's tables can
treat the methods uniformly.

Public API:

- `Solver`        — runtime-checkable protocol
- `SolveResult`   — frozen dataclass with the canonical output fields
- `SolverStatus`  — enum of (OPTIMAL, FEASIBLE, INFEASIBLE, TIMEOUT, ERROR)
- `register`      — decorator that adds a solver factory to the registry
- `get_solver`    — instantiate a registered solver by name
- `list_solvers`  — names of all registered solvers
- `validate_solution` — re-verifies budget / class-cardinality / profit
"""

from solvers.base import (
    InvalidSolutionError,
    Solver,
    SolveResult,
    SolverStatus,
    validate_solution,
)
from solvers.registry import get_solver, list_solvers, register

__all__ = [
    "InvalidSolutionError",
    "SolveResult",
    "Solver",
    "SolverStatus",
    "get_solver",
    "list_solvers",
    "register",
    "validate_solution",
]
