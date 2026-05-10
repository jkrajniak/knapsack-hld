#!/usr/bin/env python3
"""Export compact paper-facing tables from final experiment summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_FINAL_SUMMARY_DIR = Path("results") / "final_experiments" / "summary"
DEFAULT_SENSITIVITY_SUMMARY_DIR = (
    Path("results") / "final_experiments" / "time_limit_sensitivity_summary"
)
DEFAULT_OUT_DIR = Path("results") / "final_experiments" / "paper_tables"

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


def write_latex_table(path: Path, rows: list[dict[str, str]], *, columns: list[str]) -> None:
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        " \\toprule",
        " & ".join(_latex_label(column) for column in columns) + " \\\\",
        " \\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(row[column]) for column in columns) + " \\\\")
    lines.extend([" \\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _percent(value: str) -> str:
    return _decimal(str(float(value) * 100), digits=1)


def _decimal(value: str, *, digits: int) -> str:
    return f"{float(value):.{digits}f}"


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

    print(f"out_dir: {args.out_dir}")
    print(f"final_cell_status_runtime.csv: {len(final_rows)} rows")
    print(f"sensitivity_time_limit_status.csv: {len(status_rows)} rows")
    print(f"sensitivity_profit_gains.csv: {len(gain_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
