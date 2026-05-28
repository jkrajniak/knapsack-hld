r"""Emit per-N paired-comparison summary tables for §3.9 of the manuscript.

Reads ``comparison_summary/paired_profit_gaps.csv`` from a pinned
final-experiments archive and writes two LaTeX `tabular` fragments,
one per requested baseline:

* ``hld_vs_partition_summary.tex`` — paired comparison of HLD against
  the Partition-Optimal naive-decomposition baseline at each scale.
* ``hld_vs_highs_summary.tex`` — paired comparison of HLD against
  the HiGHS open-source mixed-integer reference (only available at
  N ≤ 10\,000; HiGHS does not return an incumbent at N = 100\,000).

Both fragments are designed to be ``\input{}``-ed inside an enclosing
``table`` float and to resolve the cross-references
``tab:hld_vs_partition_summary`` / ``tab:hld_vs_highs_summary`` quoted
in §3.9 of ``knapsack-optimization-paper/main.tex``.

Numbers reproduce the §3.9 prose exactly: HLD vs PO at N=100k gives
a paired median gain of +17.19% (1\,172 / 2\,046 wins); HLD vs HiGHS
at N=10k gives a paired median gap of -0.003%.

Usage:

    uv run python scripts/make_summary_tables.py \
        --paired-csv path/to/comparison_summary/paired_profit_gaps.csv \
        --out-dir results/final_experiments/paper_tables
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

BASELINES = {
    "partition_optimal": {
        "label_short": "Partition-Optimal",
        "label_long": "the Partition-Optimal naive-decomposition baseline",
        "fname": "hld_vs_partition_summary.tex",
        "expected_Ns": (1_000, 10_000, 100_000),
    },
    "highs": {
        "label_short": "HiGHS",
        "label_long": "the HiGHS open-source mixed-integer reference",
        "fname": "hld_vs_highs_summary.tex",
        "expected_Ns": (1_000, 10_000),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def _aggregate(rows: list[dict[str, str]]) -> list[dict[str, float | int]]:
    """Compute paired aggregates per N for the input subset."""
    by_N: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_N[int(row["N"])].append(float(row["hld_vs_baseline_gain_pct"]))
    out: list[dict[str, float | int]] = []
    for N in sorted(by_N):
        gains = by_N[N]
        out.append(
            {
                "N": N,
                "n": len(gains),
                "mean": statistics.mean(gains),
                "median": statistics.median(gains),
                "wins": sum(1 for g in gains if g > 0),
                "losses": sum(1 for g in gains if g < 0),
                "ties": sum(1 for g in gains if g == 0),
            }
        )
    return out


def _format_pct(value: float) -> str:
    """Match the §3.9 prose cadence: 2 decimals for |x| ≥ 1; extra precision below."""
    if abs(value) >= 1.0:
        return f"{value:+.2f}"
    if abs(value) >= 0.01:
        return f"{value:+.2f}"
    return f"{value:+.3f}"


def _format_int(value: int) -> str:
    """Format an integer with LaTeX thin-space thousands separator."""
    return f"{value:,}".replace(",", r"\,")


def _emit_table(
    aggregates: list[dict[str, float | int]],
    label_long: str,
) -> str:
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\hline",
        r"$N$ & paired $n$ & mean gain (\%) & median gain (\%) & HLD wins & "
        + r"ref wins & ties \\",
        r"\hline",
    ]
    for row in aggregates:
        N = int(row["N"])
        cells = [
            f"${_format_int(N)}$",
            _format_int(int(row["n"])),
            _format_pct(float(row["mean"])),
            _format_pct(float(row["median"])),
            _format_int(int(row["wins"])),
            _format_int(int(row["losses"])),
            _format_int(int(row["ties"])),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.paired_csv.open() as fh:
        rows = list(csv.DictReader(fh))

    for baseline, spec in BASELINES.items():
        subset = [r for r in rows if r["baseline_solver"] == baseline]
        if not subset:
            raise SystemExit(f"no rows for baseline={baseline!r} in {args.paired_csv}")
        agg = _aggregate(subset)
        seen_Ns = tuple(row["N"] for row in agg)
        if seen_Ns != spec["expected_Ns"]:
            print(
                f"warning: baseline={baseline} Ns {seen_Ns} != "
                f"expected {spec['expected_Ns']}"
            )
        out_path = args.out_dir / spec["fname"]
        out_path.write_text(_emit_table(agg, spec["label_long"]))
        n_total = sum(int(r["n"]) for r in agg)
        print(
            f"{baseline:20s} -> {out_path.name:40s} "
            f"({len(agg)} N-rows, {n_total:,} paired instances)"
        )


if __name__ == "__main__":
    main()
