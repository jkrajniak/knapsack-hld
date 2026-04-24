# `instances/` — Benchmark instance archive

Generated synthetic Selective-MCKP instances live here. The split into
**tuning** and **test** subsets is *logical* (recorded in `MANIFEST.json`)
not physical — every file lives under `<correlation>/N{N}_M{M}/...` and
the manifest tells loaders which subset each seed belongs to.

```
instances/
├── MANIFEST.json                       # SHA-256 + subset label per file
├── schema.json                         # JSON-Schema for InstanceModel
├── uncorrelated/N{N}_M{M}/mckp_*.json.gz
├── weakly/N{N}_M{M}/mckp_*.json.gz
├── strongly/N{N}_M{M}/mckp_*.json.gz
├── inversely_strongly/N{N}_M{M}/mckp_*.json.gz
└── pisinger_1995/                      # Sanity benchmark (Phase B §2.3)
```

## Generation grid

| Axis            | Values                                              |
| --------------- | --------------------------------------------------- |
| `N` (classes)   | 1 000, 10 000, 100 000                              |
| `M` (items/cls) | 5, 10, 20                                           |
| Correlation     | uncorrelated, weakly, strongly, inversely strongly  |
| `f` (tightness) | 0.1, 0.25, 0.5, 0.75, 0.9 (with `B = f · N · c̄`)   |
| Seeds           | 50 per cell (`0..49`)                               |

The generator is **deterministic** in `(N, M, correlation, f, seed)` and
files round-trip bit-exactly. Filenames follow
`mckp_N{N}_M{M}_{correlation}_f{f:0.3f}_seed{seed}.json.gz`.

## Reproducing the archive

```bash
# Tiny smoke archive (~200 instances, < 1 MB) — used in CI.
uv run python scripts/generate_instances.py \
    --config scripts/configs/archive_smoke.yaml --out instances

# Full archive (~9 000 instances, multi-GB at N=100 000).
# Expect a multi-hour run on a modern multi-core machine.
uv run python scripts/generate_instances.py \
    --config scripts/configs/archive_full.yaml --out instances

# Re-verify integrity from MANIFEST.json:
uv run python scripts/verify_instances.py --archive instances
```

## Tuning vs. test split

The split is computed by `code/instances/split.py` from a single recorded
seed (`master_seed=20260424`, `tuning_ratio=0.30`). Per cell, ~30 % of
seeds are tagged `tuning` and ~70 % `test`. All result-producing scripts
under `scripts/run_experiments/` MUST call
`instances.assert_test_only(inst)` before reporting any number; this
catches accidental leakage of tuning seeds into final results.

## Hosting strategy for large archives

The smoke archive (`scripts/configs/archive_smoke.yaml`) is committed to
the repository so CI can rebuild and verify it on every push. The full
archive (`archive_full.yaml`) is **not** committed — it would exceed
GitHub's per-file and per-repo limits. Instead it is published as a
**Zenodo release asset** alongside the tagged `v1.0-itor-revision`
release; the README and the manuscript's *Data and Code Availability
Statement* point to that DOI.

Rationale: Git LFS was considered but rejected because the bandwidth
quotas for free GitHub accounts are too restrictive (1 GB/month) and the
archive needs to be openly downloadable by reviewers without a GitHub
account.
