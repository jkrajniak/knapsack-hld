#!/usr/bin/env python3
"""Export compact paper-facing tables from final experiment summaries."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_FINAL_SUMMARY_DIR = Path("results") / "final_experiments" / "summary"
DEFAULT_SENSITIVITY_SUMMARY_DIR = (
    Path("results") / "final_experiments" / "time_limit_sensitivity_summary"
)
DEFAULT_COMPARISON_SUMMARY_DIR = Path("results") / "final_experiments" / "comparison_summary"
DEFAULT_OUT_DIR = Path("results") / "final_experiments" / "paper_tables"

PAIRED_SUMMARY_FIELDNAMES = [
    "N",
    "paired_rows",
    "hld_wins",
    "baseline_wins",
    "ties",
    "median_gain_pct",
    "mean_gain_pct",
    "median_hld_wall_time_s",
    "median_baseline_wall_time_s",
]

FINAL_CELL_FIELDNAMES = [
    "N",
    "M",
    "correlation",
    "f",
    "total_rows",
    "feasible_count",
    "timeout_count",
    "timeout_rate_pct",
    "median_wall_time_s",
    "max_wall_time_s",
]
SENSITIVITY_STATUS_FIELDNAMES = [
    "time_limit_s",
    "total_rows",
    "feasible_count",
    "timeout_count",
    "timeout_rate_pct",
]
SENSITIVITY_GAIN_FIELDNAMES = [
    "comparison",
    "n",
    "mean_gain_pct",
    "median_gain_pct",
    "max_gain_pct",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-summary-dir",
        type=Path,
        default=DEFAULT_FINAL_SUMMARY_DIR,
        help=f"Final summary directory (default: {DEFAULT_FINAL_SUMMARY_DIR}).",
    )
    parser.add_argument(
        "--sensitivity-summary-dir",
        type=Path,
        default=DEFAULT_SENSITIVITY_SUMMARY_DIR,
        help=f"Sensitivity summary directory (default: {DEFAULT_SENSITIVITY_SUMMARY_DIR}).",
    )
    parser.add_argument(
        "--comparison-summary-dir",
        type=Path,
        default=DEFAULT_COMPARISON_SUMMARY_DIR,
        help=f"Comparison summary directory (default: {DEFAULT_COMPARISON_SUMMARY_DIR}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for paper tables (default: {DEFAULT_OUT_DIR}).",
    )
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], *, fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_final_cell_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    compact_rows = [
        {
            "N": row["N"],
            "M": row["M"],
            "correlation": row["correlation"],
            "f": row["f"],
            "total_rows": row["total_rows"],
            "feasible_count": row["feasible_count"],
            "timeout_count": row["timeout_count"],
            "timeout_rate_pct": _percent(row["timeout_rate"]),
            "median_wall_time_s": _decimal(row["median_wall_time_s"], digits=2),
            "max_wall_time_s": _decimal(row["max_wall_time_s"], digits=2),
        }
        for row in rows
    ]
    return sorted(
        compact_rows,
        key=lambda row: (int(row["N"]), int(row["M"]), row["correlation"], float(row["f"])),
    )


def build_sensitivity_status_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    compact_rows = [
        {
            "time_limit_s": row["time_limit_s"],
            "total_rows": row["total_rows"],
            "feasible_count": row["feasible_count"],
            "timeout_count": row["timeout_count"],
            "timeout_rate_pct": _percent(row["timeout_rate"]),
        }
        for row in rows
    ]
    return sorted(compact_rows, key=lambda row: float(row["time_limit_s"]))


def build_sensitivity_gain_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "comparison": row["comparison"].replace("->", " to "),
            "n": row["n"],
            "mean_gain_pct": _percent(row["mean_gain"]),
            "median_gain_pct": _percent(row["median_gain"]),
            "max_gain_pct": _percent(row["max_gain"]),
        }
        for row in rows
    ]


def build_paired_summary_rows(
    rows: list[dict[str, str]],
    *,
    baseline_solver: str,
) -> list[dict[str, str]]:
    """Aggregate paired_profit_gaps.csv rows by N for a single baseline."""
    by_n: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["baseline_solver"] != baseline_solver:
            continue
        by_n[int(row["N"])].append(row)

    summary: list[dict[str, str]] = []
    for n_val in sorted(by_n):
        bucket = by_n[n_val]
        gains = [float(row["hld_vs_baseline_gain_pct"]) for row in bucket]
        hld_walls = [float(row["hld_wall_time_s"]) for row in bucket]
        base_walls = [float(row["baseline_wall_time_s"]) for row in bucket]
        wins_hld = sum(1 for g in gains if g > 0)
        wins_base = sum(1 for g in gains if g < 0)
        ties = sum(1 for g in gains if g == 0)
        summary.append(
            {
                "N": str(n_val),
                "paired_rows": str(len(bucket)),
                "hld_wins": str(wins_hld),
                "baseline_wins": str(wins_base),
                "ties": str(ties),
                "median_gain_pct": _format_gain_pct(statistics.median(gains)),
                "mean_gain_pct": _format_gain_pct(statistics.fmean(gains)),
                "median_hld_wall_time_s": _decimal(str(statistics.median(hld_walls)), digits=2),
                "median_baseline_wall_time_s": _decimal(
                    str(statistics.median(base_walls)), digits=2
                ),
            }
        )
    return summary


def write_latex_table(
    path: Path,
    rows: list[dict[str, str]],
    *,
    columns: list[str],
    caption: str | None = None,
) -> None:
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        " \\toprule",
        " & ".join(_latex_label(column) for column in columns) + " \\\\",
        " \\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(row[column]) for column in columns) + " \\\\")
    lines.extend([" \\bottomrule", "\\end{tabular}"])
    if caption is not None:
        lines.insert(0, f"% {caption}")
    lines.append("")
    path.write_text("\n".join(lines))


# Manuscript-facing paired-summary tables (\ref{tab:hld_vs_partition_summary},
# \ref{tab:hld_vs_highs_summary}, §3.9). Column ordering, math-wrapped N,
# \,-grouped thousands, and \hline rules match the rest of main.tex by
# convention so the manuscript layout stays consistent across regenerations.
PAIRED_MANUSCRIPT_COLUMNS = [
    "N",
    "paired_rows",
    "mean_gain_pct",
    "median_gain_pct",
    "hld_wins",
    "baseline_wins",
    "ties",
]
PAIRED_MANUSCRIPT_HEADERS = [
    "$N$",
    "paired $n$",
    "mean gain (\\%)",
    "median gain (\\%)",
    "HLD wins",
    "ref wins",
    "ties",
]


def _format_thousands(value: str) -> str:
    """Render an integer with LaTeX \\, thousands separators (e.g. 2100 → 2\\,100)."""
    n = int(value)
    sign = "-" if n < 0 else ""
    digits = str(abs(n))
    groups = []
    while digits:
        groups.append(digits[-3:])
        digits = digits[:-3]
    return sign + "\\,".join(reversed(groups))


def write_paired_summary_latex(
    path: Path,
    rows: list[dict[str, str]],
    *,
    caption: str,
) -> None:
    """Emit paired-summary table in the manuscript's table style.

    Differs from `write_latex_table` (booktabs, generic column spec): uses
    right-aligned `rrrrrrr` columns, `\\hline` rules, math-wrapped `$N$`,
    and `\\,`-grouped thousands to match the surrounding §3 tables.
    """
    n_cols = len(PAIRED_MANUSCRIPT_COLUMNS)
    lines = [
        f"% {caption}",
        "\\begin{tabular}{" + "r" * n_cols + "}",
        "\\hline",
        " & ".join(PAIRED_MANUSCRIPT_HEADERS) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        rendered = [
            "$" + _format_thousands(row["N"]) + "$",
            _format_thousands(row["paired_rows"]),
            row["mean_gain_pct"],
            row["median_gain_pct"],
            _format_thousands(row["hld_wins"]),
            _format_thousands(row["baseline_wins"]),
            _format_thousands(row["ties"]),
        ]
        lines.append(" & ".join(rendered) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _percent(value: str) -> str:
    return _decimal(str(float(value) * 100), digits=1)


def _decimal(value: str, *, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def _format_gain_pct(value: float | str) -> str:
    """Format a percentage already expressed in percent units, with sign.

    Project number-cadence convention (PI L152 ruling 2026-05-26):
    percentages with |x| >= 1 reported to 2 decimals; sub-1 % keep 4
    decimals to stay informative when 2 decimals would round to zero.
    """
    x = float(value)
    if abs(x) < 1.0:
        return f"{x:+.4f}"
    return f"{x:+.2f}"


def _latex_label(value: str) -> str:
    labels = {
        "time_limit_s": "Time limit (s)",
        "total_rows": "Rows",
        "feasible_count": "Feasible",
        "timeout_count": "Timeout",
        "timeout_rate_pct": "Timeout (\\%)",
        "comparison": "Comparison",
        "n": "n",
        "mean_gain_pct": "Mean gain (\\%)",
        "median_gain_pct": "Median gain (\\%)",
        "max_gain_pct": "Max gain (\\%)",
        "paired_rows": "Paired rows",
        "hld_wins": "HLD wins",
        "baseline_wins": "Baseline wins",
        "ties": "Ties",
        "median_hld_wall_time_s": "Median HLD time (s)",
        "median_baseline_wall_time_s": "Median baseline time (s)",
    }
    return labels.get(value, _latex_escape(value))


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    final_rows = build_final_cell_rows(read_csv(args.final_summary_dir / "cell_summary.csv"))
    status_rows = build_sensitivity_status_rows(
        read_csv(args.sensitivity_summary_dir / "time_limit_status.csv")
    )
    gain_rows = build_sensitivity_gain_rows(
        read_csv(args.sensitivity_summary_dir / "profit_gains.csv")
    )

    paired_csv = args.comparison_summary_dir / "paired_profit_gaps.csv"
    paired_available = paired_csv.exists()
    if paired_available:
        paired_rows = read_csv(paired_csv)
        po_summary = build_paired_summary_rows(paired_rows, baseline_solver="partition_optimal")
        highs_summary = build_paired_summary_rows(paired_rows, baseline_solver="highs")
    else:
        po_summary = []
        highs_summary = []

    write_csv(
        args.out_dir / "final_cell_status_runtime.csv",
        final_rows,
        fieldnames=FINAL_CELL_FIELDNAMES,
    )
    write_csv(
        args.out_dir / "sensitivity_time_limit_status.csv",
        status_rows,
        fieldnames=SENSITIVITY_STATUS_FIELDNAMES,
    )
    write_csv(
        args.out_dir / "sensitivity_profit_gains.csv",
        gain_rows,
        fieldnames=SENSITIVITY_GAIN_FIELDNAMES,
    )
    write_latex_table(
        args.out_dir / "sensitivity_time_limit_status.tex",
        status_rows,
        columns=SENSITIVITY_STATUS_FIELDNAMES,
    )
    write_latex_table(
        args.out_dir / "sensitivity_profit_gains.tex",
        gain_rows,
        columns=SENSITIVITY_GAIN_FIELDNAMES,
    )
    if paired_available:
        write_csv(
            args.out_dir / "hld_vs_partition_summary.csv",
            po_summary,
            fieldnames=PAIRED_SUMMARY_FIELDNAMES,
        )
        write_csv(
            args.out_dir / "hld_vs_highs_summary.csv",
            highs_summary,
            fieldnames=PAIRED_SUMMARY_FIELDNAMES,
        )
        write_paired_summary_latex(
            args.out_dir / "hld_vs_partition_summary.tex",
            po_summary,
            caption="Paired HLD vs Partition-Optimal aggregated by N",
        )
        write_paired_summary_latex(
            args.out_dir / "hld_vs_highs_summary.tex",
            highs_summary,
            caption="Paired HLD vs HiGHS aggregated by N",
        )

    print(f"out_dir: {args.out_dir}")
    print(f"final_cell_status_runtime.csv: {len(final_rows)} rows")
    print(f"sensitivity_time_limit_status.csv: {len(status_rows)} rows")
    print(f"sensitivity_profit_gains.csv: {len(gain_rows)} rows")
    if paired_available:
        print(f"hld_vs_partition_summary.csv: {len(po_summary)} rows")
        print(f"hld_vs_highs_summary.csv: {len(highs_summary)} rows")
    else:
        print(f"hld_vs_partition_summary.csv: skipped ({paired_csv} not found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
