"""Solver registry — name → factory.

Solvers register themselves at import time via the `@register("name")`
decorator. Benchmark scripts and the SMAC3 harness then look them up by
string name, which keeps the call sites declarative and makes it
trivial to add or swap baselines.
"""

from __future__ import annotations

from collections.abc import Callable

from solvers.base import Solver

_FACTORIES: dict[str, Callable[[], Solver]] = {}


def register(name: str) -> Callable[[Callable[[], Solver]], Callable[[], Solver]]:
    """Decorator that adds a zero-argument factory under `name`."""
    if not name:
        raise ValueError("solver name must be non-empty")

    def _wrap(factory: Callable[[], Solver]) -> Callable[[], Solver]:
        if name in _FACTORIES:
            raise ValueError(f"solver name already registered: {name!r}")
        _FACTORIES[name] = factory
        return factory

    return _wrap


def get_solver(name: str) -> Solver:
    """Instantiate the solver registered under `name`."""
    if name not in _FACTORIES:
        raise KeyError(f"unknown solver {name!r}; known: {sorted(_FACTORIES)}")
    solver = _FACTORIES[name]()
    if solver.name != name:
        raise RuntimeError(
            f"factory for {name!r} returned a solver named {solver.name!r}; "
            "names must match the registry key"
        )
    return solver


def list_solvers() -> list[str]:
    """Return the sorted list of registered solver names."""
    return sorted(_FACTORIES)


def _snapshot_and_clear() -> dict[str, Callable[[], Solver]]:
    """Test-only hook: save the current registry, then clear it."""
    snap = dict(_FACTORIES)
    _FACTORIES.clear()
    return snap


def _restore(snapshot: dict[str, Callable[[], Solver]]) -> None:
    """Test-only hook: restore a registry snapshot."""
    _FACTORIES.clear()
    _FACTORIES.update(snapshot)
