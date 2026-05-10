# Experiment Follow-Ups

This note records follow-up analyses identified during the final HLD experiment
run. The current run should continue to completion; these checks are intended
for interpretation and robustness analysis after the first full CSV is written.

## 1. Status and Runtime Summary

Create a cell-level summary from `results/final_experiments/results.csv`:

- Group by `N`, `M`, `correlation`, `f`, and `solver`.
- Count statuses: `optimal`, `feasible`, `timeout`, and `error`.
- Report runtime statistics: mean, median, maximum, and selected quantiles.
- Report profit statistics per cell so timeout-heavy cells can be compared with
  cells that usually finish before the time limit.

This summary should make the final tables transparent about how often HLD hit
the configured wall-clock cap.

## 2. Time-Limit Sensitivity Analysis

Use the status summary to select cells with many `timeout` rows, then rerun a
small representative subset with several HLD time limits, for example:

- 30 seconds
- 60 seconds
- 120 seconds
- 300 seconds

Compare profit, status distribution, and runtime across the limits. The goal is
to check whether the 60-second cap materially changes conclusions on the hardest
cells, without rerunning the full benchmark grid at every time limit.
