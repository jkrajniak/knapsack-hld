# SMAC3 Tuning of HLD — Preview Report

> **Status:** preview run on the small N=200 tuning subset (this branch
> ships only 16 tuning instances across the four correlation kinds and
> two M values). The full 5 000-trial campaign is gated on the full
> benchmark archive.
>
> The harness, parameter space, target metric, safety hooks, and
> bootstrap CI logic are exactly the same as in the planned full run;
> only the tuning subset and the `--budget` differ.

## Summary

|                                          | Mean optimality gap | 95 % CI (bootstrap)   | Mean runtime | 95 % CI         |
|------------------------------------------|--------------------:|-----------------------|-------------:|------------------|
| **Default** (manuscript §2.7)            | 2.43 %              | [1.52 %, 3.55 %]      | 2.10 s       | [0.98, 3.48]     |
| **SMAC incumbent** (preview, this run)   | **1.45 %**          | [0.97 %, 1.98 %]      | **1.54 s**   | [0.62, 3.06]     |

The SMAC incumbent's 95 % gap CI sits **below** the default's mean,
which on this tuning subset is consistent with a genuine (non-noise)
improvement; both bootstrap CIs use the same 16 instances.

## Recommended configuration (preview)

| Parameter | Manuscript default | SMAC incumbent | Search range |
|-----------|-------------------:|---------------:|-------------:|
| `N_iter`     | 20    | **23**    | [5, 50]    |
| `α`          | 0.90  | **0.123** | [0.0, 1.0] |
| `K`          | 8     | **6**     | [4, 64]    |
| `λ_max`      | 10.0  | **12.50** | [1.0, 100.0] |

**Headline finding (preview).** SMAC moves α from the manuscript's
0.9 toward equal allocation (α ≈ 0.12). On the inversely-correlated
N=200 cells in this preview subset, the Phase-1 selection-cost
estimates that drive the proportional component of the budget
allocation are noisy enough that an almost-equal split outperforms
proportional allocation.

This is **directly relevant to the α=0.9 vs α=1.0 question**.
The full-archive campaign should resolve whether this
preview-scale finding generalises to the N=10 000 / 100 000 regimes —
or whether it is an artefact of the tiny preview subset (16 instances,
all `inversely_strongly`).

## Reproducibility

```bash
cd /path/to/knapsack-hld
PYTHONPATH=code uv run python -m tuning.smac_run \
    --preview --budget 150 --seed 7 --max-instances 16 \
    --eval-time-limit-s 30 --bootstrap-resamples 1000
```

| Artefact                                      | Description                                  |
|-----------------------------------------------|----------------------------------------------|
| `preview/incumbent.json`                      | Recommended config + bootstrap 95 % CI       |
| `preview/comparison.json`                     | Default vs incumbent on identical instances  |
| `preview/evaluations.csv`                     | Per-trial `(config, instance, gap, time)`    |
| `preview/reference_profits.json`              | HiGHS oracle cache                           |
| `preview/hld_smac/7/runhistory.json`          | Full SMAC run history (auto-written by SMAC) |
| `preview/hld_smac/7/scenario.json`            | SMAC scenario snapshot                       |
| `preview/hld_smac/7/configspace.json`         | ConfigSpace definition                       |

| Provenance | Value |
|---|---|
| `git_sha`         | `7b5e39b` (preview run) |
| `smac_seed`       | `7` |
| `n_trials_total`  | `150` |
| `tuning_subset`   | 16 of 80 manifest entries with `subset == "tuning"` |
| `archive`         | `instances/` (preview, N=200 only) |

Every loaded instance is asserted to be in the **tuning** subset by
`instances.split.assert_tuning_only` inside the SMAC target callback;
attempts to load test-subset instances raise `AssertionError`.

## Method (one paragraph)

SMAC3 (`AlgorithmConfigurationFacade`) explores the four-dimensional
HLD parameter space `(N_iter, α, K, λ_max)` over the tuning subset of
the instance manifest, optimising mean optimality gap relative to a
HiGHS reference profit. Scenario is `deterministic=True` with a fixed
SMAC seed; the target callback uses `assert_tuning_only` so the
campaign cannot leak signal from the held-out test partition. The
incumbent is re-evaluated on every tuning instance for an unbiased
bootstrap-percentile 95 % CI (1 000 resamples). HiGHS reference
profits are cached on disk so multi-stage and re-runs are fast.

## Caveats and gates

- **Preview scale only.** This run uses the 16 N=200 tuning instances
  shipping with the change branch. The full N ∈ {1 k, 10 k, 100 k}
  archive (§2.2.1) is gated on the user-side multi-hour generator run.
- **All preview instances are `inversely_strongly` correlated** (the
  alphabetically-first cell). This is the hardest correlation kind, so
  the preview is biased toward "robustness against poor Phase-1
  estimates" — likely the reason α drops so far below 0.9.
- **Manuscript updates (§4.2.4) are deferred** until the full-archive
  campaign confirms (or refutes) the preview finding on N=10 000 and
  N=100 000.
- **SMAC instance features are not used yet** (SMAC emits a soft
  warning). For the preview the surrogate model treats all instances
  as equivalent; for the full campaign, adding `(N, M, correlation, f)`
  one-hot features would let SMAC condition on cell.

## Spec coverage

| Task | Spec scenario | Status (preview) |
|------|----------------|------------------|
| §4.1.1 | Parameter space `(N_iter, α, K, λ_max)` exposed to SMAC | ✅ `code/tuning/smac_run.py:PARAM_SPACE` |
| §4.1.2 | `AlgorithmConfigurationFacade`, deterministic seed, budget cap | ✅ `--budget`, `--seed`, `deterministic=True` |
| §4.1.3 | Runtime assertion every loaded instance is in tuning subset | ✅ `assert_tuning_only` in target callback |
| §4.2.1 | Archive `runhistory.json`, `scenario.json`, `incumbent.json` | ✅ all three present in `preview/hld_smac/7/` and `preview/` |
| §4.2.2 | Bootstrap 95 % CI on incumbent | ✅ `tuning.bootstrap.bootstrap_mean_ci`, in `incumbent.json` |
| §4.2.3 | One-page report comparing recommended vs default | ✅ this file |
| §4.2.4 | Update §2.7 / §3.4 / §3.5 in manuscript | ⏳ pending full-archive run |
