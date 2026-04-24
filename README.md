# knapsack-hld

Reference implementation and reproducibility package for the
**Hybrid Lagrangian-Decomposition (HLD) algorithm** for the
*Selective Multiple-Choice Knapsack Problem* (Selective-MCKP).

This repository accompanies the manuscript

> Krajniak, J. *A Hybrid Lagrangian-Decomposition Algorithm for the Large-Scale
> Multiple-Choice Knapsack Problem.* International Transactions in Operational
> Research (revision under review, 2026).

The package is intentionally **open-source-only**: every solver, heuristic,
benchmark generator, and tuning component runs without commercial licences
(no Gurobi, no CPLEX, no R dependencies). Open-source MILP solvers
(HiGHS, SCIP, COIN-OR CBC) are used for both the HLD inner solves and the
exact baselines.

## Status

🚧 Skeleton — Phase A (week 1) of the ITOR major revision. Implementation of
benchmark suite, baselines, tuning campaign and HLD instrumentation lands in
Phases B–F (see `openspec/changes/itor-major-revision-2026/tasks.md` in the
companion paper repository).

## Quick start

```bash
# Requires Python ≥ 3.12 and uv (https://docs.astral.sh/uv/)
uv sync
uv run pytest
```

End-to-end reproduction of every figure and table in the manuscript will be
available via:

```bash
make reproduce            # full archive (long; runs in parallel)
make reproduce-quick      # smoke subset (< 10 min)
```

## Repository layout

```
knapsack-hld/
├── code/
│   ├── instances/   # Pure-Python MCKP instance generator (4 correlation classes)
│   ├── solvers/     # Unified wrapper for HiGHS, SCIP, CBC, mcknap, etc.
│   ├── heuristics/  # Greedy-MaxRatio, BISSA, TRS-2008, Partition-Optimal
│   ├── hld/         # Hybrid Lagrangian-Decomposition algorithm
│   ├── tuning/      # SMAC3 tuning campaign
│   └── utils/       # Shared metrics, IO, logging, parallel primitives
├── instances/       # Generated benchmark archive + MANIFEST.json
├── scripts/         # End-user CLI (make reproduce, generate, run, plot, …)
├── results/         # Raw experimental output (gzipped CSV)
├── tuning/          # SMAC3 run history and chosen incumbent
├── figures/         # PDF figures used by the paper
├── paper/           # LaTeX source mirror (optional)
└── tests/           # Unit and integration tests
```

## Citation

A Zenodo concept DOI is reserved for the v1.0 release; the camera-ready
version of the paper will cite the exact tag used to produce the published
results.

## License

MIT — see [`LICENSE`](LICENSE).
