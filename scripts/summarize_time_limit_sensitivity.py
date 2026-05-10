#!/usr/bin/env python3
"""Summarize HLD time-limit sensitivity results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_SENSITIVITY_CSV = Path("results") / "final_experiments" / "time_limit_sensitivity.csv"
DEFAULT_OUT_DIR = Path("results") / "final_experiments" / "time_limit_sensitivity_summary"
DEFAULT_COMPARISONS = (("30", "60"), ("60", "120"), ("120", "300"), ("60", "300"))

STATUS_FIELDNAMES = [
    "time_limit_s",
    "total_rows",
    "status_counts",
    "feasible_count",
    "timeout_count",
    "timeout_rate",
]
GAIN_FIELDNAMES = ["comparison", "n", "mean_gain", "median_gain", "max_gain"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensitivity-csv",
        type=Path,
        default=DEFAULT_SENSITIVITY_CSV,
        help=f"Sensitivity CSV path (default: {DEFAULT_SENSITIVITY_CSV}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for summary files (default: {DEFAULT_OUT_DIR}).",
    )
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def build_overall_summary(rows: list[dict[str, str]], *, source: Path) -> dict[str, Any]:
    return {
        "source": str(source),
        "total_rows": len(rows),
        "time_limits_s": sorted(
            {row["time_limit_s"] for row in rows if row.get("time_limit_s")},
            key=float,
        ),
        "status_counts": dict(sorted(Counter(row.get("status", "") for row in rows).items())),
    }


def build_status_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["time_limit_s"]].append(row)

    summary_rows = []
    for time_limit_s, limit_rows in grouped.items():
        statuses = Counter(row.get("status", "") for row in limit_rows)
        timeout_count = statuses.get("timeout", 0)
        total_rows = len(limit_rows)
        summary_rows.append(
            {
                "time_limit_s": time_limit_s,
                "total_rows": str(total_rows),
                "status_counts": _format_counts(statuses),
                "feasible_count": str(statuses.get("feasible", 0)),
                "timeout_count": str(timeout_count),
                "timeout_rate": _format_float(timeout_count / total_rows if total_rows else 0.0),
            }
        )
    return sorted(summary_rows, key=lambda row: float(row["time_limit_s"]))


def build_profit_gain_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_instance = _group_by_instance(rows)
    return [
        _summarize_profit_gain(by_instance, lower=lower, upper=upper)
        for lower, upper in DEFAULT_COMPARISONS
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, str]], *, fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_by_instance(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (row["instance_id"], row["N"], row["M"], row["correlation"], row["f"])
        grouped[key][row["time_limit_s"]] = row
    return grouped


def _summarize_profit_gain(
    by_instance: dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]],
    *,
    lower: str,
    upper: str,
) -> dict[str, str]:
    gains = []
    for values in by_instance.values():
        if lower not in values or upper not in values:
            continue
        lower_profit = _to_int(values[lower].get("profit", ""))
        upper_profit = _to_int(values[upper].get("profit", ""))
        if lower_profit is None or upper_profit is None or lower_profit <= 0:
            continue
        gains.append((upper_profit - lower_profit) / lower_profit)

    gains.sort()
    return {
        "comparison": f"{lower}->{upper}",
        "n": str(len(gains)),
        "mean_gain": _format_float(mean(gains) if gains else 0.0),
        "median_gain": _format_float(median(gains) if gains else 0.0),
        "max_gain": _format_float(max(gains) if gains else 0.0),
    }


def _format_counts(counts: Counter[str]) -> str:
    return ";".join(f"{status}={count}" for status, count in sorted(counts.items()) if status)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _to_int(value: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.sensitivity_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    status_rows = build_status_summary(rows)
    gain_rows = build_profit_gain_summary(rows)
    write_json(
        args.out_dir / "overall.json", build_overall_summary(rows, source=args.sensitivity_csv)
    )
    write_csv(args.out_dir / "time_limit_status.csv", status_rows, fieldnames=STATUS_FIELDNAMES)
    write_csv(args.out_dir / "profit_gains.csv", gain_rows, fieldnames=GAIN_FIELDNAMES)

    print(f"sensitivity_csv: {args.sensitivity_csv}")
    print(f"out_dir: {args.out_dir}")
    print(f"total_rows: {len(rows)}")
    print("status_by_time_limit:")
    for row in status_rows:
        print(f"{row['time_limit_s']}: {row['status_counts']}")
    print("profit_gains:")
    for row in gain_rows:
        print(
            f"{row['comparison']}: n={row['n']} "
            f"mean={float(row['mean_gain']):.4%} "
            f"median={float(row['median_gain']):.4%} "
            f"max={float(row['max_gain']):.4%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
