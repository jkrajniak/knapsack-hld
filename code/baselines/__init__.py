"""Pure-Python re-implementations of MCKP / Selective-MCKP baselines.

Modules under `baselines/` are deliberately separated from `solvers/`
because they are *re-implementations of published algorithms* whose
correctness is itself part of the paper's claim. Each module owns its
algorithmic core, its provenance, and its own README.

Currently registered through the unified `solvers` interface:

- `mcknap`   — exact MCKP solver in the spirit of Pisinger 1995
              (Sinha-Zoltners LP relaxation + branch-and-bound)
- `trs2008`  — Tsesmetzis-Roussaki-Sykas 2008 two-phase greedy
              Selective-MCKP heuristic
"""

from baselines import mcknap as _mcknap
from baselines import trs2008 as _trs2008

__all__ = ["_mcknap", "_trs2008"]
