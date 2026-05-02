# Baselines (pure-Python)

Re-implementations of MCKP / Selective-MCKP baselines whose correctness
is itself part of the paper's reproducibility claim. Every baseline is
exposed through the unified `solvers` interface; see `code/solvers/`.

## `mcknap.py` — exact MCKP / Selective-MCKP solver

### Algorithm

Two stages:

1. **Sinha-Zoltners (1979) LP relaxation** on the **concave upper hull**
   of each class. We keep two views of every class:
   - `pareto`   — all Pareto-optimal items (used for B&B branching)
   - `envelope` — strict concave upper hull (used for the LP increments)

2. **Branch-and-bound** that branches on the LP cracking class. We
   branch over the **full Pareto set** of the cracking class so no
   integer-optimal item is ever missed; the LP bound prunes against
   the incumbent.

For Selective-MCKP, every class is augmented internally with a virtual
`(0, 0)` dummy item; the dummy is mapped back to `None` in the final
`SolveResult.items_selected`. This is the standard MCKP→Selective-MCKP
transformation.

### Differences from Pisinger 1995

Pisinger's "minimal algorithm" uses a primal-dual *core* approach plus
a tightly-tuned branch-and-bound around a small core of classes, and
gradient-projection refinement. Our implementation is intentionally
simpler:

- **No core extraction.** We branch on the full Pareto set of the
  cracking class.
- **No gradient projection.** B&B alone closes the gap.
- **Pure Python.** Numpy is used only for sorting in the envelope build.

The reasoning is as follows:

> A faithful pure-Python implementation of Pisinger's primal-dual core
> algorithm is high-risk for low marginal benefit (the reviewers want
> correctness on Pisinger's archive, not algorithmic identity). A clean
> LP+B&B in the spirit of Pisinger 1995 returns the same optimum on
> every Pisinger archive instance, is auditable end-to-end without
> external C dependencies, and is sufficient as a sanity baseline.

### Performance characteristics

LP+B&B is asymptotically the same complexity class as Pisinger's
algorithm but has a larger constant on hard correlation classes
(particularly `inversely_strongly`). On `N=12`, `M=4` it solves any
instance in `<1 s`. On `N=50+`, `M=5` instances of `inversely_strongly`
correlation it can take minutes. Heavy use of `mcknap` is intended for
the **Pisinger 1995 archive sanity check** and **§5-style worked
examples**, not for the manuscript's main very-large-N benchmarks
(which use HiGHS / SCIP / CBC / HLD).

### Tests

- `tests/test_mcknap_pisinger1995_example.py`
  - hand-crafted small MCKP with hand-verifiable optimum
  - 20-instance correctness gate vs HiGHS across all four correlations
  - Selective-MCKP class-skipping
  - graceful timeout

### Pisinger archive validation

`tests/test_mcknap_pisinger1995_archive.py` (TODO when the user has
fetched the archive per `instances/pisinger_1995/README.md`) runs
mcknap on every Pisinger 1995 archive instance and confirms it matches
the published optimum (or HiGHS as a proxy when the published optimum
is unavailable).
