#!/usr/bin/env python3
"""Compare completed HLD final results against baseline result CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_HLD_CSV = Path("results") / "final_experiments" / "results.csv"
DEFAULT_OUT_DIR = Path("results") / "final_experiments" / "comparison_summary"

STATUS_FIELDNAMES = [
    "solver",
    "N",
    "M",
    "correlation",
    "f",
    "total_rows",
    "status_counts",
    "feasible_count",
    "timeout_count",
    "optimal_count",
    "error_count",
    "median_wall_time_s",
    "max_wall_time_s",
]
PAIRED_FIELDNAMES = [
    "baseline_solver",
    "instance_id",
    "N",
    "M",
    "correlation",
    "f",
    "seed",
    "hld_status",
    "baseline_status",
    "hld_profit",
    "baseline_profit",
    "hld_vs_baseline_gain_pct",
    "hld_wall_time_s",
    "baseline_wall_time_s",
    "winner",
]
AGGREGATE_FIELDNAMES = [
    "baseline_solver",
    "N",
    "M",
    "correlation",
    "f",
    "paired_rows",
    "hld_wins",
    "baseline_wins",
    "ties",
    "mean_hld_vs_baseline_gain_pct",
    "median_hld_vs_baseline_gain_pct",
    "max_hld_vs_baseline_gain_pct",
    "median_hld_wall_time_s",
    "median_baseline_wall_time_s",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hld-csv",
        type=Path,
        default=DEFAULT_HLD_CSV,
        help=f"Completed HLD final result CSV (default: {DEFAULT_HLD_CSV}).",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more baseline result CSVs to compare against HLD.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for comparison summaries (default: {DEFAULT_OUT_DIR}).",
    )
    return parser.parse_args(argv)


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]], *, fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_status_runtime_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_solver_cell_key(row)].append(row)

    summaries = []
    for key, group_rows in grouped.items():
        solver, n_items, n_classes, correlation, f_value = key
        statuses = Counter(row.get("status", "") for row in group_rows)
        wall_times = [_to_float(row.get("wall_time_s", "")) for row in group_rows]
        valid_wall_times = [value for value in wall_times if value is not None]
        summaries.append(
            {
                "solver": solver,
                "N": n_items,
                "M": n_classes,
                "correlation": correlation,
                "f": f_value,
                "total_rows": str(len(group_rows)),
                "status_counts": _format_counts(statuses),
                "feasible_count": str(statuses.get("feasible", 0)),
                "timeout_count": str(statuses.get("timeout", 0)),
                "optimal_count": str(statuses.get("optimal", 0)),
                "error_count": str(statuses.get("error", 0)),
                "median_wall_time_s": _format_float(
                    median(valid_wall_times) if valid_wall_times else 0.0
                ),
                "max_wall_time_s": _format_float(
                    max(valid_wall_times) if valid_wall_times else 0.0
                ),
            }
        )
    return sorted(summaries, key=lambda row: _sort_key(row, include_solver=True))


def build_paired_rows(
    hld_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    hld_by_instance = {row["instance_id"]: row for row in hld_rows if row.get("instance_id")}
    pairs = []
    for baseline_row in baseline_rows:
        instance_id = baseline_row.get("instance_id", "")
        if instance_id not in hld_by_instance:
            continue
        hld_row = hld_by_instance[instance_id]
        hld_profit = _to_int(hld_row.get("profit", ""))
        baseline_profit = _to_int(baseline_row.get("profit", ""))
        if hld_profit is None or baseline_profit is None or baseline_profit <= 0:
            continue
        pairs.append(
            {
                "baseline_solver": baseline_row["solver"],
                "instance_id": instance_id,
                "N": hld_row["N"],
                "M": hld_row["M"],
                "correlation": hld_row["correlation"],
                "f": hld_row["f"],
                "seed": hld_row["seed"],
                "hld_status": hld_row["status"],
                "baseline_status": baseline_row["status"],
                "hld_profit": str(hld_profit),
                "baseline_profit": str(baseline_profit),
                "hld_vs_baseline_gain_pct": _format_float(
                    100.0 * (hld_profit - baseline_profit) / baseline_profit
                ),
                "hld_wall_time_s": _format_float(_to_float(hld_row.get("wall_time_s", "")) or 0.0),
                "baseline_wall_time_s": _format_float(
                    _to_float(baseline_row.get("wall_time_s", "")) or 0.0
                ),
                "winner": _winner(hld_profit, baseline_profit),
            }
        )
    return sorted(
        pairs,
        key=lambda row: (
            row["baseline_solver"],
            int(row["N"]),
            int(row["M"]),
            row["correlation"],
            float(row["f"]),
            row["instance_id"],
        ),
    )


def build_aggregate_rows(paired_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in paired_rows:
        grouped[(row["baseline_solver"], row["N"], row["M"], row["correlation"], row["f"])].append(
            row
        )

    aggregate_rows = []
    for key, rows in grouped.items():
        solver, n_items, n_classes, correlation, f_value = key
        gains = [float(row["hld_vs_baseline_gain_pct"]) for row in rows]
        hld_times = [float(row["hld_wall_time_s"]) for row in rows]
        baseline_times = [float(row["baseline_wall_time_s"]) for row in rows]
        winners = Counter(row["winner"] for row in rows)
        aggregate_rows.append(
            {
                "baseline_solver": solver,
                "N": n_items,
                "M": n_classes,
                "correlation": correlation,
                "f": f_value,
                "paired_rows": str(len(rows)),
                "hld_wins": str(winners.get("hld", 0)),
                "baseline_wins": str(winners.get("baseline", 0)),
                "ties": str(winners.get("tie", 0)),
                "mean_hld_vs_baseline_gain_pct": _format_float(mean(gains)),
                "median_hld_vs_baseline_gain_pct": _format_float(median(gains)),
                "max_hld_vs_baseline_gain_pct": _format_float(max(gains)),
                "median_hld_wall_time_s": _format_float(median(hld_times)),
                "median_baseline_wall_time_s": _format_float(median(baseline_times)),
            }
        )
    return sorted(aggregate_rows, key=lambda row: _sort_key(row, include_solver=True))


def write_latex_table(path: Path, rows: list[dict[str, str]], *, fieldnames: list[str]) -> None:
    lines = [
        "\\begin{tabular}{" + "l" * len(fieldnames) + "}",
        " \\toprule",
        " & ".join(_latex_label(field) for field in fieldnames) + " \\\\",
        " \\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(row[field]) for field in fieldnames) + " \\\\")
    lines.extend([" \\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _solver_cell_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (row["solver"], row["N"], row["M"], row["correlation"], row["f"])


def _sort_key(row: dict[str, str], *, include_solver: bool) -> tuple[Any, ...]:
    base = (int(row["N"]), int(row["M"]), row["correlation"], float(row["f"]))
    if include_solver:
        return (row.get("solver") or row.get("baseline_solver"), *base)
    return base


def _winner(hld_profit: int, baseline_profit: int) -> str:
    if hld_profit > baseline_profit:
        return "hld"
    if baseline_profit > hld_profit:
        return "baseline"
    return "tie"


def _format_counts(counts: Counter[str]) -> str:
    return ";".join(f"{status}={count}" for status, count in sorted(counts.items()) if status)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _to_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _latex_label(value: str) -> str:
    labels = {
        "baseline_solver": "Baseline",
        "paired_rows": "Pairs",
        "hld_wins": "HLD wins",
        "baseline_wins": "Baseline wins",
        "mean_hld_vs_baseline_gain_pct": "Mean gain (\\%)",
        "median_hld_vs_baseline_gain_pct": "Median gain (\\%)",
        "median_hld_wall_time_s": "Median HLD time (s)",
        "median_baseline_wall_time_s": "Median baseline time (s)",
    }
    return labels.get(value, _latex_escape(value))


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hld_rows = load_rows([args.hld_csv])
    baseline_rows = load_rows(args.baseline_csv)
    all_rows = [*hld_rows, *baseline_rows]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    status_rows = build_status_runtime_rows(all_rows)
    paired_rows = build_paired_rows(hld_rows, baseline_rows)
    aggregate_rows = build_aggregate_rows(paired_rows)

    write_json(
        args.out_dir / "overall.json",
        {
            "hld_csv": str(args.hld_csv),
            "baseline_csvs": [str(path) for path in args.baseline_csv],
            "hld_rows": len(hld_rows),
            "baseline_rows": len(baseline_rows),
            "paired_rows": len(paired_rows),
        },
    )
    write_csv(
        args.out_dir / "solver_status_runtime.csv",
        status_rows,
        fieldnames=STATUS_FIELDNAMES,
    )
    write_csv(
        args.out_dir / "paired_profit_gaps.csv",
        paired_rows,
        fieldnames=PAIRED_FIELDNAMES,
    )
    write_csv(
        args.out_dir / "aggregate_profit_gaps.csv",
        aggregate_rows,
        fieldnames=AGGREGATE_FIELDNAMES,
    )
    write_latex_table(
        args.out_dir / "aggregate_profit_gaps.tex",
        aggregate_rows,
        fieldnames=[
            "baseline_solver",
            "N",
            "M",
            "correlation",
            "f",
            "paired_rows",
            "hld_wins",
            "baseline_wins",
            "median_hld_vs_baseline_gain_pct",
        ],
    )

    print(f"out_dir: {args.out_dir}")
    print(f"solver_status_runtime.csv: {len(status_rows)} rows")
    print(f"paired_profit_gaps.csv: {len(paired_rows)} rows")
    print(f"aggregate_profit_gaps.csv: {len(aggregate_rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
