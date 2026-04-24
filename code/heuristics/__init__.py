"""Polynomial-time heuristic baselines for Selective-MCKP.

Heuristics live here (rather than under `baselines/`) because their
algorithmic recipe is a couple of lines, not a re-implementation of a
published exact procedure. Every heuristic registers itself through the
unified `solvers` registry so benchmark scripts treat it like any other
solver.

Currently registered:

- `greedy_max_ratio`  — sort all items by profit/cost; pick greedily.
- `partition_optimal` — split classes into K equal-sized batches, solve
                        each batch with an exact MILP (B/K per batch).
- `bissa`             — Bednarczuk, Miroforidis & Pyzel (2018) bi-objective
                        scalarisation, with the explicit
                        MCKP -> Selective-MCKP dummy-item transformation
                        recorded in `solver_metadata`.

`Greedy-MaxProfit` was the original "pick highest-profit item that fits"
heuristic from the v1 manuscript. It is *intentionally* NOT registered
here — see `tests/test_greedy_max_profit_removed.py` and the rationale
in `README.md`.
"""

from heuristics import bissa as _bissa
from heuristics import greedy as _greedy
from heuristics import partition_optimal as _partition_optimal

__all__ = ["_bissa", "_greedy", "_partition_optimal"]
