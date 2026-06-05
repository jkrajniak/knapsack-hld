#!/usr/bin/env python3
"""Regenerate manuscript figures from pinned-archive CSVs.

Scaffold: each figure is a small named generator (`FIGURE_GENERATORS`)
that reads a known CSV from the pinned final-experiments archive and
emits a PDF plus a sibling `<name>.meta.json` provenance sidecar.

Provenance sidecar (per Task 3.2.2 of revision-finalization-2026):
    {
      "figure": "<name>",
      "script": "scripts/make_figures.py",
      "command": "<argv that produced this figure>",
      "generated_at": "<ISO-8601 UTC>",
      "archive": {"id": "<archive-basename>", "sha256": "<sha256>"},
      "source_csv": "<path>",
      "n_paired": <int>,
      "stats": {...}
    }

Status:
    - hld_vs_partition_paired_gains: implemented (§3.9 pivot-aligned)
    - others: TODO; see paper/FIGURE_NOTES.md in knapsack-research for the
      open scope decisions.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shlex
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_PAIRED_CSV = (
    Path("results")
    / "final_experiments"
    / "comparison_summary"
    / "paired_profit_gaps.csv"
)
DEFAULT_RESULTS_CSV = Path("results") / "final_experiments" / "results.csv"
DEFAULT_LAMBDA_SWEEP = Path("results") / "anomalies" / "full" / "sweep.jsonl"
DEFAULT_ALPHA_SWEEP = Path("results") / "anomalies" / "full_alpha" / "sweep.jsonl"
DEFAULT_OUT_DIR = Path("figures")

# Plotted solvers for `large_scale_scaling_ab` — pivot-aligned subset.
SCALING_SOLVERS = ["hld", "partition_optimal", "highs"]
SCALING_LABEL = {
    "hld": "HLD",
    "partition_optimal": "Partition-Optimal",
    "highs": "HiGHS (reference)",
}
SCALING_COLOR = {
    "hld": "tab:blue",
    "partition_optimal": "tab:orange",
    "highs": "tab:green",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-csv",
        type=Path,
        default=DEFAULT_PAIRED_CSV,
        help=f"comparison_summary/paired_profit_gaps.csv (default: {DEFAULT_PAIRED_CSV}).",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        action="append",
        default=None,
        help="Raw per-instance results CSV. Repeatable; when omitted, defaults "
        f"to {DEFAULT_RESULTS_CSV} only (HLD). In the pinned archive layout, "
        "Partition-Optimal lives in partition_optimal_refreshed.csv and "
        "HiGHS in highs_baseline_maxN10000.csv; pass all three to get the "
        "full HLD / PO / HiGHS scaling figure.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for figure PDFs and meta sidecars (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--archive-id",
        type=str,
        default="",
        help="Basename of the pinned archive these CSVs come from "
        "(e.g. final_experiments_20260524T202545Z_partition_refreshed_v2.tar.gz).",
    )
    parser.add_argument(
        "--archive-sha256",
        type=str,
        default="",
        help="SHA-256 of the pinned archive (records provenance).",
    )
    parser.add_argument(
        "--lambda-sweep-jsonl",
        type=Path,
        default=DEFAULT_LAMBDA_SWEEP,
        help=f"JSONL sweep for lambda_sensitivity figure (default: {DEFAULT_LAMBDA_SWEEP}).",
    )
    parser.add_argument(
        "--alpha-sweep-jsonl",
        type=Path,
        default=DEFAULT_ALPHA_SWEEP,
        help=f"JSONL sweep for allocation_ratio_sensitivity figure (default: {DEFAULT_ALPHA_SWEEP}).",
    )
    parser.add_argument(
        "--only",
        type=str,
        action="append",
        default=None,
        help="Generate only this named figure (repeatable). Default: all.",
    )
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _seed_from_inst_id(inst_id: str) -> int:
    """Extract seed integer from instance id like '..._seed0' / '..._seed12'."""
    marker = "_seed"
    idx = inst_id.rfind(marker)
    if idx < 0:
        raise ValueError(f"inst_id missing '_seed' marker: {inst_id!r}")
    return int(inst_id[idx + len(marker):])


def _group_by_x_seed(
    rows: list[dict], *, x_key: str, y_keys: list[str]
) -> tuple[list[float], dict[int, dict[float, dict[str, float]]]]:
    """Group sweep rows by x value and seed.

    Returns (sorted_x_values, {seed: {x: {y_key: value, ...}}}).
    """
    by_seed: dict[int, dict[float, dict[str, float]]] = {}
    x_values: set[float] = set()
    for row in rows:
        seed = _seed_from_inst_id(row["inst_id"])
        x = float(row[x_key])
        x_values.add(x)
        by_seed.setdefault(seed, {})[x] = {key: float(row[key]) for key in y_keys}
    return sorted(x_values), by_seed


def _mean_sd_by_x(
    by_seed: dict[int, dict[float, dict[str, float]]],
    *,
    x_values: list[float],
    y_key: str,
) -> tuple[list[float], list[float]]:
    """Compute mean and population SD across seeds for each x value."""
    means: list[float] = []
    sds: list[float] = []
    for x in x_values:
        per_seed = [by_seed[seed][x][y_key] for seed in sorted(by_seed) if x in by_seed[seed]]
        means.append(statistics.fmean(per_seed))
        sds.append(statistics.pstdev(per_seed) if len(per_seed) > 1 else 0.0)
    return means, sds


def _make_sensitivity_figure(
    args: argparse.Namespace,
    *,
    figure_name: str,
    source_jsonl: Path,
    x_key: str,
    x_label: str,
    selected_marker: tuple[float, str] | None,
) -> None:
    """Render a two-panel sensitivity figure (quality + wall time) from a JSONL sweep.

    Plots per-seed lines plus a mean line; gap is expressed as percent.
    `selected_marker` is an optional (x_value, label) to mark with a vertical
    dashed line on both panels (e.g. the chosen $N_{\text{iter}} = 20$).
    """
    rows = _read_jsonl(source_jsonl)
    x_values, by_seed = _group_by_x_seed(
        rows, x_key=x_key, y_keys=["optimality_gap", "hld_wall_s"]
    )
    seeds = sorted(by_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out_dir / f"{figure_name}.pdf"

    fig, (ax_q, ax_t) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    for seed in seeds:
        seed_xs = sorted(by_seed[seed])
        gaps_pct = [by_seed[seed][x]["optimality_gap"] * 100.0 for x in seed_xs]
        walls = [by_seed[seed][x]["hld_wall_s"] for x in seed_xs]
        ax_q.plot(seed_xs, gaps_pct, marker=".", alpha=0.4, linewidth=0.8,
                  label=f"seed {seed}")
        ax_t.plot(seed_xs, walls, marker=".", alpha=0.4, linewidth=0.8,
                  label=f"seed {seed}")

    mean_gap_pct = [v * 100.0 for v in
                    _mean_sd_by_x(by_seed, x_values=x_values, y_key="optimality_gap")[0]]
    mean_wall, _ = _mean_sd_by_x(by_seed, x_values=x_values, y_key="hld_wall_s")
    ax_q.plot(x_values, mean_gap_pct, color="black", linewidth=1.8, label="mean")
    ax_t.plot(x_values, mean_wall, color="black", linewidth=1.8, label="mean")

    if selected_marker is not None:
        x_sel, label_sel = selected_marker
        for ax in (ax_q, ax_t):
            ax.axvline(x_sel, color="tab:red", linestyle="--", linewidth=1.0,
                       alpha=0.7, label=label_sel)

    ax_q.set_xlabel(x_label)
    ax_q.set_ylabel("Optimality gap (\\%)")
    ax_q.set_title("(a) Solution quality")
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="best", fontsize=8)

    ax_t.set_xlabel(x_label)
    ax_t.set_ylabel("HLD wall time (s)")
    ax_t.set_title("(b) Computation time")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)

    inst_ids = sorted({row["inst_id"] for row in rows})
    stats = {
        "n_rows": len(rows),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "x_values": x_values,
        "instances": inst_ids,
        "mean_gap_pct_per_x": dict(zip([str(x) for x in x_values], mean_gap_pct, strict=True)),
        "mean_wall_s_per_x": dict(zip([str(x) for x in x_values], mean_wall, strict=True)),
    }
    meta = {
        "figure": figure_name,
        "script": "scripts/make_figures.py",
        "command": shlex.join(sys.argv),
        "generated_at": _now_iso(),
        "archive": {"id": args.archive_id, "sha256": args.archive_sha256},
        "source_jsonl": str(source_jsonl),
        "stats": stats,
        "notes": (
            "Single-cell anomaly sweep: hardest weakly-correlated cell "
            "(N=10000, M=10, weakly correlated, f=0.5) across 3 seeds "
            "(see stats.seeds for actual seed values). Quality panel shows "
            "per-seed gap traces plus their across-seed mean; time panel "
            "mirrors the same structure for HLD wall time."
        ),
    }
    meta_path = pdf_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def make_lambda_sensitivity(args: argparse.Namespace) -> None:
    """§3.5 (R.4 regen): lambda search iterations $N_{\\text{iter}}$ sensitivity.

    Two-panel: (a) optimality gap vs $N_{\\text{iter}}$, (b) HLD wall time
    vs $N_{\\text{iter}}$. Source: pinned anomalies/full/sweep.jsonl (75 rows
    = 3 seeds × 25 $N_{\\text{iter}}$ values on a single anomaly cell).
    """
    _make_sensitivity_figure(
        args,
        figure_name="lambda_sensitivity",
        source_jsonl=args.lambda_sweep_jsonl,
        x_key="n_iter",
        x_label="$N_{\\mathrm{iter}}$ (lambda bisection iterations)",
        selected_marker=(20.0, "selected $N_{\\mathrm{iter}}=20$"),
    )


def make_allocation_ratio_sensitivity(args: argparse.Namespace) -> None:
    """§3.6 (R.5 regen): allocation-ratio $\\alpha$ sensitivity.

    Two-panel: (a) optimality gap vs $\\alpha$, (b) HLD wall time vs
    $\\alpha$. Source: pinned anomalies/full_alpha/sweep.jsonl (33 rows
    = 3 seeds × 11 $\\alpha$ values, with $N_{\\text{iter}}$ fixed at 20).
    """
    _make_sensitivity_figure(
        args,
        figure_name="allocation_ratio_sensitivity",
        source_jsonl=args.alpha_sweep_jsonl,
        x_key="alpha",
        x_label="Allocation ratio $\\alpha$",
        selected_marker=(0.9, "selected $\\alpha=0.9$"),
    )


def _read_paired_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _write_meta(
    pdf_path: Path,
    *,
    figure: str,
    args: argparse.Namespace,
    source_csv: Path,
    n_paired: int,
    stats: dict[str, float | int],
) -> None:
    meta = {
        "figure": figure,
        "script": "scripts/make_figures.py",
        "command": shlex.join(sys.argv),
        "generated_at": _now_iso(),
        "archive": {"id": args.archive_id, "sha256": args.archive_sha256},
        "source_csv": str(source_csv),
        "n_paired": n_paired,
        "stats": stats,
    }
    meta_path = pdf_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def make_hld_vs_partition_paired_gains(args: argparse.Namespace) -> None:
    """ECDF of paired HLD-vs-Partition-Optimal profit gains at N=100 000.

    Source: comparison_summary/paired_profit_gaps.csv filtered to
    baseline_solver=partition_optimal and N=100000. Vertical line at 0 %.
    """
    figure_name = "hld_vs_partition_paired_gains"
    rows = _read_paired_rows(args.paired_csv)
    gains = sorted(
        float(row["hld_vs_baseline_gain_pct"])
        for row in rows
        if row["baseline_solver"] == "partition_optimal" and int(row["N"]) == 100_000
    )
    if not gains:
        raise SystemExit(
            f"no partition_optimal rows at N=100000 in {args.paired_csv}; "
            "is the archive contents-correct?"
        )

    n = len(gains)
    ecdf_y = [(i + 1) / n for i in range(n)]
    wins = sum(1 for g in gains if g > 0)
    losses = sum(1 for g in gains if g < 0)
    ties = sum(1 for g in gains if g == 0)
    stats = {
        "n_paired": n,
        "hld_wins": wins,
        "baseline_wins": losses,
        "ties": ties,
        "median_gain_pct": statistics.median(gains),
        "mean_gain_pct": statistics.fmean(gains),
        "min_gain_pct": gains[0],
        "max_gain_pct": gains[-1],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out_dir / f"{figure_name}.pdf"

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.step(gains, ecdf_y, where="post", linewidth=1.4, color="tab:blue")
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=0.8)
    median = stats["median_gain_pct"]
    ax.axvline(median, color="tab:red", linestyle=":", linewidth=0.8)
    ax.annotate(
        f"median = {median:+.2f}%",
        xy=(median, 0.5),
        xytext=(8, 0),
        textcoords="offset points",
        color="tab:red",
        fontsize=9,
        va="center",
    )
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("HLD vs Partition-Optimal profit gain (%)")
    ax.set_ylabel("Cumulative fraction of paired instances")
    ax.set_title(
        f"HLD vs Partition-Optimal at $N=100{{,}}000$ "
        f"(n={n}, HLD wins={wins}/{n})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)

    _write_meta(
        pdf_path,
        figure=figure_name,
        args=args,
        source_csv=args.paired_csv,
        n_paired=n,
        stats=stats,
    )


def _aggregate_by_solver_n(
    rows: list[dict[str, str]],
    solvers: list[str],
) -> dict[str, dict[int, dict[str, float | int]]]:
    """Group rows by solver -> N and compute median wall time and profit."""
    by_solver_n: dict[str, dict[int, list[dict[str, str]]]] = {s: {} for s in solvers}
    for row in rows:
        solver = row["solver"]
        if solver not in by_solver_n:
            continue
        n = int(row["N"])
        by_solver_n[solver].setdefault(n, []).append(row)

    out: dict[str, dict[int, dict[str, float | int]]] = {}
    for solver, by_n in by_solver_n.items():
        if not by_n:
            continue
        out[solver] = {}
        for n, bucket in sorted(by_n.items()):
            walls = [float(r["wall_time_s"]) for r in bucket]
            profits = [float(r["profit"]) for r in bucket]
            out[solver][n] = {
                "n": len(bucket),
                "median_wall_time_s": statistics.median(walls),
                "median_profit": statistics.median(profits),
            }
    return out


def make_large_scale_scaling_ab(args: argparse.Namespace) -> None:
    """Two-panel Fig 9 replacement: (a) wall time vs N, (b) profit vs N.

    Plots the pivot-aligned solver subset {HLD, Partition-Optimal,
    HiGHS reference} from the pinned final_experiments archive.
    Replaces the legacy four-panel large_scale_validation figure;
    panels (c) memory and (d) efficiency were dropped per PI D5
    ruling 2026-05-27 (Task 3.2.4).
    """
    figure_name = "large_scale_scaling_ab"
    results_csvs = args.results_csv or [DEFAULT_RESULTS_CSV]
    rows: list[dict[str, str]] = []
    for path in results_csvs:
        with path.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))

    feasible_rows = [row for row in rows if row["status"] != "error"]
    agg = _aggregate_by_solver_n(feasible_rows, SCALING_SOLVERS)
    if not agg:
        raise SystemExit(
            f"no rows for solvers {SCALING_SOLVERS} in {[str(p) for p in results_csvs]}; "
            "is the archive contents-correct?"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out_dir / f"{figure_name}.pdf"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for solver in SCALING_SOLVERS:
        if solver not in agg:
            continue
        ns = sorted(agg[solver])
        walls = [agg[solver][n]["median_wall_time_s"] for n in ns]
        profits = [agg[solver][n]["median_profit"] for n in ns]
        ax_a.plot(
            ns,
            walls,
            marker="o",
            color=SCALING_COLOR[solver],
            label=SCALING_LABEL[solver],
        )
        ax_b.plot(
            ns,
            profits,
            marker="o",
            color=SCALING_COLOR[solver],
            label=SCALING_LABEL[solver],
        )

    for ax, title, ylabel in (
        (ax_a, "(a) Median wall time", "Median wall time (s)"),
        (ax_b, "(b) Median profit", "Median profit"),
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of decision classes $N$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)

    stats = {
        solver: {str(n): per_n for n, per_n in by_n.items()}
        for solver, by_n in agg.items()
    }
    meta = {
        "figure": figure_name,
        "script": "scripts/make_figures.py",
        "command": shlex.join(sys.argv),
        "generated_at": _now_iso(),
        "archive": {"id": args.archive_id, "sha256": args.archive_sha256},
        "source_csv": [str(p) for p in results_csvs],
        "solvers": [s for s in SCALING_SOLVERS if s in agg],
        "stats": stats,
    }
    meta_path = pdf_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def make_average_computation_time(args: argparse.Namespace) -> None:
    """Fig 2 (manuscript): mean wall time vs N for HLD, Partition-Optimal, HiGHS.

    Replaces the legacy `average_computation_time.pdf` asset that lacked a
    provenance sidecar. Reads any combination of per-instance result CSVs
    (pinned final-experiments + small-N rerun) via `--results-csv`,
    aggregates mean and median wall time per (solver, N), and writes a
    single-panel log-log line plot plus a `.meta.json` sidecar.

    HiGHS at N=50000 is intentionally excluded from the line: HiGHS model
    preprocessing alone exceeds the 60 s wall budget at that size (see
    `results_highs_N50000_preprocessing_cliff_*.csv` for the negative-
    result evidence). The caption is expected to call this out.
    """
    figure_name = "average_computation_time"
    results_csvs = args.results_csv or [DEFAULT_RESULTS_CSV]
    rows: list[dict[str, str]] = []
    for path in results_csvs:
        with path.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))

    plotted_rows = [
        row
        for row in rows
        if row["solver"] in SCALING_SOLVERS and row["status"] != "error"
    ]
    agg = _aggregate_by_solver_n(plotted_rows, SCALING_SOLVERS)
    if not agg:
        raise SystemExit(
            f"no rows for solvers {SCALING_SOLVERS} in {[str(p) for p in results_csvs]};"
            " is the archive contents-correct?"
        )

    mean_wall = {
        solver: {
            n: statistics.fmean(
                float(r["wall_time_s"])
                for r in plotted_rows
                if r["solver"] == solver and int(r["N"]) == n
            )
            for n in by_n
        }
        for solver, by_n in agg.items()
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out_dir / f"{figure_name}.pdf"

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for solver in SCALING_SOLVERS:
        if solver not in agg:
            continue
        ns = sorted(agg[solver])
        means = [mean_wall[solver][n] for n in ns]
        ax.plot(
            ns,
            means,
            marker="o",
            color=SCALING_COLOR[solver],
            label=SCALING_LABEL[solver],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of decision classes $N$")
    ax.set_ylabel("Mean wall time (s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)

    stats = {
        solver: {
            str(n): {
                "n_instances": agg[solver][n]["n"],
                "mean_wall_time_s": mean_wall[solver][n],
                "median_wall_time_s": agg[solver][n]["median_wall_time_s"],
            }
            for n in sorted(agg[solver])
        }
        for solver in agg
    }
    meta = {
        "figure": figure_name,
        "script": "scripts/make_figures.py",
        "command": shlex.join(sys.argv),
        "generated_at": _now_iso(),
        "archive": {"id": args.archive_id, "sha256": args.archive_sha256},
        "source_csv": [str(p) for p in results_csvs],
        "solvers": [s for s in SCALING_SOLVERS if s in agg],
        "stats": stats,
        "notes": (
            "HiGHS at N=50000 is excluded: model preprocessing alone "
            "exceeds the 60 s per-instance wall budget; see "
            "results_highs_N50000_preprocessing_cliff_*.csv for the "
            "negative-result evidence (64 instances, all profit=0, "
            "median wall 660 s)."
        ),
    }
    meta_path = pdf_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


FIGURE_GENERATORS: dict[str, Callable[[argparse.Namespace], None]] = {
    "hld_vs_partition_paired_gains": make_hld_vs_partition_paired_gains,
    "large_scale_scaling_ab": make_large_scale_scaling_ab,
    "average_computation_time": make_average_computation_time,
    "lambda_sensitivity": make_lambda_sensitivity,
    "allocation_ratio_sensitivity": make_allocation_ratio_sensitivity,
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    requested = args.only if args.only else list(FIGURE_GENERATORS.keys())
    unknown = [name for name in requested if name not in FIGURE_GENERATORS]
    if unknown:
        raise SystemExit(f"unknown figure(s): {', '.join(unknown)}")
    for name in requested:
        FIGURE_GENERATORS[name](args)
        print(f"generated: {args.out_dir / f'{name}.pdf'} (+ .meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
