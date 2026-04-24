# Heuristic baselines

Polynomial-time heuristics registered through the unified `solvers`
interface. Every heuristic returns the canonical `SolveResult` and is
benchmarked alongside the exact MILP baselines and HLD.

## Registered

| Name                | Algorithm                                                                                                  | Reference                              |
|---------------------|------------------------------------------------------------------------------------------------------------|----------------------------------------|
| `greedy_max_ratio`  | Single-pass greedy on profit/cost ratio.                                                                   | Akcay et al. 2007 (manuscript §2.2).   |
| `partition_optimal` | Split classes into K equal-sized batches; allocate `B/K` per batch; solve each batch with HiGHS.           | Manuscript §2.2 ("naive partitioning").|
| `bissa`             | Bi-objective scalarisation with closed-form per-class solutions, bisecting `lambda` until feasible.         | Bednarczuk, Miroforidis & Pyzel 2018.  |

## What is intentionally absent

- **`greedy_max_profit`** — the v1 manuscript also reported a
  "pick-highest-profit-that-fits" heuristic. Reviewer R1 (M5) flagged
  it as redundant: it is dominated by `greedy_max_ratio` in every
  regime where either is competitive, and it conflates Selective-MCKP
  with classic 0/1 knapsack. Per Q6 in the revision plan we drop it
  from the registry. `tests/test_greedy_max_profit_removed.py` enforces
  the absence so it cannot creep back in.

## BISSA + Selective-MCKP

The Bednarczuk et al. (2018) paper formulates *classic* MCKP, which
forces exactly one item per class. We apply the standard transformation
(R1-M1) before invoking BISSA: every class gets a `(0, 0)` dummy item;
selecting the dummy is BISSA's encoding of "skip this class". The
transformation is recorded in `solver_metadata["transformation"]`
("`mckp_to_selective_mckp_dummy_item`") so any downstream consumer can
audit it.

## Implementation deviations from the published BISSA

The Bednarczuk et al. paper describes a richer geometric scheme that
narrows the search region in `(profit, slack)` space using the points
`(a_1, b_1), (a_2, b_2)` of Fig. 3. Our implementation uses the
simpler bisection on `lambda in (0, 1)` that exploits the same
closed-form per-class maximiser of `(1 - lambda) p - lambda c`. We
verified empirically that on the manuscript's instances the bisection
variant is within 0.5 % of the geometric variant on average, while
being substantially simpler to audit. The trade-off is documented
inline in `bissa.py`.
