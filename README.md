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

This repository is an active revision package. It includes the benchmark
generator, solver adapters, HLD implementation, tuning harnesses, anomaly
analysis scripts, and reproducibility checks needed for the revised
manuscript.

## Quick start

```bash
# Requires Python ≥ 3.12 and uv (https://docs.astral.sh/uv/)
uv sync
uv run pytest
```

## Reproducing the benchmark archive

The steps below are also wrapped as `make` targets — run `make help` to list
them (`make reproduce-quick`, `make reproduce`, `make test`, `make lint`).

A quick smoke build — the same flow run in CI (< 10 min) — generates a small
instance archive and verifies its manifest:

```bash
uv run python scripts/generate_instances.py \
    --config scripts/configs/archive_smoke.yaml --out instances --jobs 2
uv run python scripts/verify_instances.py --archive instances
```

The full benchmark archive (long; runs in parallel) is driven by:

```bash
scripts/run_full_archive.sh --out instances_full_candidate --jobs 16
```

The manuscript tables and figures are regenerated from the produced `results/`
CSVs with the `scripts/make_*_tables.py` and `scripts/make_figures.py` helpers.

## Guarded HLD (`guarded_hld`)

When the correlation class is unknown at deployment time, use the registered
solver **`guarded_hld`** instead of raw `hld`. It is a thin wrapper (not a new
algorithm): run Partition-Optimal at the operational batch count, optionally
skip full HLD when the Lagrangian upper bound is within `τ_skip` of PO
(default `0.005`, cost-only), otherwise run HLD and return **`max(PO, HLD)`**
so profit never falls below the equal-budget baseline.

```bash
# Example: batch-granularity sweep cell (see scripts/configs/batch_granularity_*.yaml)
uv run python scripts/check_batch_granularity.py \
  --config scripts/configs/batch_granularity_midN.yaml \
  --methods guarded_hld partition_optimal hld
```

Implementation: `code/solvers/guarded_hld.py`. Use raw `hld` only when the
instance is known heterogeneous *and* batch granularity is fine enough that
unguarded allocation error is acceptable (see manuscript §4.8).

## Repository layout

```
knapsack-hld/
├── code/
│   ├── instances/   # Pure-Python MCKP instance generator (4 correlation classes)
│   ├── solvers/     # Unified wrapper for HiGHS, SCIP, CBC, mcknap, etc.
│   ├── baselines/   # Exact-baseline and reference solver drivers
│   ├── heuristics/  # Greedy-MaxRatio, BISSA, TRS-2008, Partition-Optimal
│   ├── hld/         # Hybrid Lagrangian-Decomposition algorithm
│   ├── tuning/      # SMAC3 tuning campaign
│   ├── anomalies/   # Anomaly-detection analysis helpers
│   └── utils/       # Shared metrics, IO, logging, parallel primitives
├── instances/       # Generated benchmark archive + MANIFEST.json
├── scripts/         # End-user CLI (generate, run, plot, archive, …)
├── results/         # Raw experimental output (gzipped CSV)
├── tuning/          # SMAC3 run history and chosen incumbent
├── figures/         # PDF figures used by the paper
├── configs/         # Chosen SMAC incumbents (JSON)
├── docs/            # Supplementary notes
├── paper/           # Manuscript source mirror (see paper/README.md)
└── tests/           # Unit and integration tests
```

## Citation

A Zenodo concept DOI is reserved for the v1.0 release; the camera-ready
version of the paper will cite the exact tag used to produce the published
results.

## License

MIT — see [`LICENSE`](LICENSE).
