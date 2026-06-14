# HLD anomaly investigation (Phase D §4.3.2)

Mechanistic check of two hypotheses for the figure-anomaly behaviour:

- **H1 — degenerate dual basis.** Phase-1 fails to converge: `lambda_rel_spread > 1%` over the final iterations *or* ≥ 3 sign flips of `total_cost - B`.
- **H2 — sub-MILP straggler.** Slowest Phase-3 batch consumes ≥ 50% of total Phase-3 wall time.

## Summary

- Records analysed: **9** (1 instance(s) x 9 N_iter values)
- Mean optimality gap: **0.22%**
- Mean HLD wall time: **88.33 s**
- H1 (degenerate dual) flagged: **67%** of records
- H2 (sub-MILP straggler) flagged: **0%** of records


## Per-instance trajectories


### `mckp_N10000_M10_weakly_f0.500_seed0`

| N_iter | gap | HLD wall (s) | λ_final | gap to B | λ-spread | sign-flips | max-batch share | H1 | H2 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 1 | 1.66% | 73.14 | 50.000 | -2500485 | 0.000 | 0 | 25% | · | · |
| 2 | 0.15% | 83.60 | 25.000 | -2500359 | 0.500 | 0 | 24% | ✗ | · |
| 3 | 0.18% | 63.92 | 12.500 | -2499826 | 0.750 | 0 | 28% | ✗ | · |
| 5 | 0.02% | 114.49 | 3.125 | -2482893 | 0.938 | 0 | 30% | ✗ | · |
| 8 | 0.00% | 88.44 | 1.172 | -1161879 | 0.875 | 2 | 21% | ✗ | · |
| 12 | -0.00% | 77.28 | 1.099 | -62748 | 0.167 | 2 | 26% | ✗ | · |
| 16 | -0.00% | 110.76 | 1.094 | +23544 | 0.011 | 3 | 22% | ✗ | · |
| 20 | 0.00% | 94.06 | 1.095 | -897 | 0.001 | 1 | 26% | · | · |
| 25 | 0.00% | 89.24 | 1.095 | -897 | 0.000 | 2 | 27% | · | · |

## Interpretation

- **H1 supported.** Phase-1 routinely fails to converge on this anomaly subset: the Lagrange multiplier oscillates and the selection cost flips around the budget. The Phase-2 estimated costs ``C_k`` are therefore noisy, which is consistent with the non-monotonic gap-vs-N_iter behaviour reported in Figure 4.
- H2 not supported by this run. Phase-3 wall time is balanced across batches.

