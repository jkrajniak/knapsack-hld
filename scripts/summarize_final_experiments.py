#!/usr/bin/env python3
"""Summarize final experiment result CSVs for review analyses."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_RESULTS_CSV = Path("results") / "final_experiments" / "results.csv"
DEFAULT_OUT_DIR = Path("results") / "final_experiments" / "summary"

CELL_FIELDNAMES = [
    "solver",
    "N",
    "M",
    "correlation",
    "f",
    "total_rows",
    "status_counts",
    "feasible_count",
    "timeout_count",
    "error_count",
    "timeout_rate",
    "mean_wall_time_s",
    "median_wall_time_s",
    "max_wall_time_s",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"Final experiment CSV path (default: {DEFAULT_RESULTS_CSV}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for summary files (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--top-timeout-cells",
        type=int,
        default=25,
        help="Maximum timeout-heavy cells to write (default: 25).",
    )
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CELL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_overall_summary(rows: list[dict[str, str]], *, source: Path) -> dict[str, Any]:
    wall_times = [_to_float(row.get("wall_time_s", "")) for row in rows]
    valid_wall_times = [value for value in wall_times if value is not None]
    return {
        "source": str(source),
        "total_rows": len(rows),
        "solvers": sorted({row.get("solver", "") for row in rows if row.get("solver")}),
        "status_counts": dict(sorted(Counter(row.get("status", "") for row in rows).items())),
        "wall_time_s": _summarize_numbers(valid_wall_times),
    }


def build_cell_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, int, int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_cell_key(row)].append(row)

    summaries = [_summarize_cell(key, cell_rows) for key, cell_rows in grouped.items()]
    return sorted(
        summaries,
        key=lambda row: (
            row["solver"],
            int(row["N"]),
            int(row["M"]),
            row["correlation"],
            float(row["f"]),
        ),
    )


def top_timeout_cells(cell_rows: list[dict[str, str]], *, limit: int) -> list[dict[str, str]]:
    timeout_rows = [row for row in cell_rows if int(row["timeout_count"]) > 0]
    ordered = sorted(
        timeout_rows,
        key=lambda row: (
            -float(row["timeout_rate"]),
            -int(row["timeout_count"]),
            -int(row["total_rows"]),
            row["solver"],
            int(row["N"]),
            int(row["M"]),
            row["correlation"],
            float(row["f"]),
        ),
    )
    return ordered[:limit]


def _cell_key(row: dict[str, str]) -> tuple[str, int, int, str, str]:
    return (
        row.get("solver", ""),
        int(row.get("N", 0)),
        int(row.get("M", 0)),
        row.get("correlation", ""),
        row.get("f", ""),
    )


def _summarize_cell(
    key: tuple[str, int, int, str, str],
    rows: list[dict[str, str]],
) -> dict[str, str]:
    solver, n_items, n_classes, correlation, f_value = key
    statuses = Counter(row.get("status", "") for row in rows)
    wall_times = [
        value
        for value in (_to_float(row.get("wall_time_s", "")) for row in rows)
        if value is not None
    ]
    total_rows = len(rows)
    timeout_count = statuses.get("timeout", 0)
    return {
        "solver": solver,
        "N": str(n_items),
        "M": str(n_classes),
        "correlation": correlation,
        "f": f_value,
        "total_rows": str(total_rows),
        "status_counts": _format_counts(statuses),
        "feasible_count": str(statuses.get("feasible", 0)),
        "timeout_count": str(timeout_count),
        "error_count": str(statuses.get("error", 0)),
        "timeout_rate": _format_float(timeout_count / total_rows if total_rows else 0.0),
        "mean_wall_time_s": _format_float(mean(wall_times) if wall_times else 0.0),
        "median_wall_time_s": _format_float(median(wall_times) if wall_times else 0.0),
        "max_wall_time_s": _format_float(max(wall_times) if wall_times else 0.0),
    }


def _summarize_numbers(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "total": 0.0, "min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, int(0.95 * (len(sorted_values) - 1)))
    return {
        "count": len(sorted_values),
        "total": sum(sorted_values),
        "min": sorted_values[0],
        "median": median(sorted_values),
        "p95": sorted_values[p95_index],
        "max": sorted_values[-1],
    }


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.results_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cell_rows = build_cell_summary(rows)
    timeout_rows = top_timeout_cells(cell_rows, limit=args.top_timeout_cells)

    write_json(args.out_dir / "overall.json", build_overall_summary(rows, source=args.results_csv))
    write_csv(args.out_dir / "cell_summary.csv", cell_rows)
    write_csv(args.out_dir / "top_timeout_cells.csv", timeout_rows)

    print(f"results_csv: {args.results_csv}")
    print(f"out_dir: {args.out_dir}")
    print(f"total_rows: {len(rows)}")
    print(f"cell_summary_rows: {len(cell_rows)}")
    print(f"top_timeout_cells: {len(timeout_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
