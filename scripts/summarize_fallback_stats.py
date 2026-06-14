#!/usr/bin/env python3
"""Aggregate HLD Phase-2 fallback statistics (Task 3.4.1).

Reads a `results/final_experiments/results.csv` produced by
`run_final_experiments.py` (post-3.4.1, with the `fallback_equal_split`
column) and emits:

- `fallback_stats.csv`        -- per-cell fallback counts and rates
- `fallback_overall.json`     -- overall + per-N summary

Rows where `solver != hld` or `fallback_equal_split` is empty are
ignored. This makes the script backward-compatible with CSVs predating
the Task 3.4.1 schema change (they simply yield `n_rows = 0`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cell:
    n_items: int
    n_classes: int
    correlation: str
    f_value: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        type=Path,
        required=True,
        help="Path to the final experiments results.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for fallback_stats.csv + fallback_overall.json.",
    )
    return parser.parse_args(argv)


def _interpret_flag(raw: str | None) -> int | None:
    """Map the CSV value to {0, 1} or None if absent."""
    if raw is None:
        return None
    s = raw.strip().lower()
    if s == "":
        return None
    if s in {"1", "true", "yes"}:
        return 1
    if s in {"0", "false", "no"}:
        return 0
    return None


def load_hld_fallback_rows(path: Path) -> list[tuple[Cell, int]]:
    out: list[tuple[Cell, int]] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if "fallback_equal_split" not in (reader.fieldnames or []):
            return out
        for row in reader:
            if row.get("solver") != "hld":
                continue
            flag = _interpret_flag(row.get("fallback_equal_split"))
            if flag is None:
                continue
            out.append(
                (
                    Cell(
                        n_items=int(row["N"]),
                        n_classes=int(row["M"]),
                        correlation=row["correlation"],
                        f_value=float(row["f"]),
                    ),
                    flag,
                )
            )
    return out


def per_cell_stats(rows: list[tuple[Cell, int]]) -> list[dict[str, object]]:
    buckets: dict[Cell, list[int]] = defaultdict(list)
    for cell, flag in rows:
        buckets[cell].append(flag)
    return [
        {
            "N": cell.n_items,
            "M": cell.n_classes,
            "correlation": cell.correlation,
            "f": cell.f_value,
            "n_rows": len(flags),
            "n_fallbacks": sum(flags),
            "fallback_rate": (sum(flags) / len(flags)) if flags else 0.0,
        }
        for cell, flags in sorted(
            buckets.items(),
            key=lambda kv: (kv[0].n_items, kv[0].n_classes, kv[0].correlation, kv[0].f_value),
        )
    ]


def overall_stats(rows: list[tuple[Cell, int]]) -> dict[str, object]:
    by_n: dict[int, list[int]] = defaultdict(list)
    for cell, flag in rows:
        by_n[cell.n_items].append(flag)
    total = [flag for _, flag in rows]
    return {
        "n_rows": len(total),
        "n_fallbacks": sum(total),
        "fallback_rate": (sum(total) / len(total)) if total else 0.0,
        "by_N": {
            str(n): {
                "n_rows": len(flags),
                "n_fallbacks": sum(flags),
                "fallback_rate": (sum(flags) / len(flags)) if flags else 0.0,
            }
            for n, flags in sorted(by_n.items())
        },
    }


def write_per_cell_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["N", "M", "correlation", "f", "n_rows", "n_fallbacks", "fallback_rate"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_hld_fallback_rows(args.results_csv)
    overall = overall_stats(rows)
    per_cell = per_cell_stats(rows)
    write_per_cell_csv(args.out_dir / "fallback_stats.csv", per_cell)
    (args.out_dir / "fallback_overall.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n"
    )
    print(f"results_csv: {args.results_csv}")
    print(f"out_dir: {args.out_dir}")
    print(f"n_rows: {overall['n_rows']}")
    print(f"n_fallbacks: {overall['n_fallbacks']}")
    print(f"fallback_rate: {overall['fallback_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
