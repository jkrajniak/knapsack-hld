#!/usr/bin/env python3
"""Summarise the class-ordering ablation (Task 3.3.2).

Reads per-ordering CSVs written by `run_final_experiments.py` (one
file per ordering, e.g. `sequential.csv`, `random.csv`,
`adversarial.csv`), joins them by `instance_id`, and emits
per-ordering and per-cell gap statistics for §3.6 of the revision.

The gap metric is **paired**: for each instance, the best profit
across the three orderings is the reference, and each ordering's
gap is `(best - own) / best * 100`. Ties contribute 0 gap to every
ordering involved; an ordering that wins counts toward its
`win_rate` (multiple orderings can co-win on the same instance).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ORDERINGS: tuple[str, ...] = ("sequential", "random", "adversarial")


@dataclass(frozen=True)
class Row:
    instance_id: str
    cell: tuple[int, int, str, float]
    profit: float | None
    status: str
    wall_time_s: float
    ordering: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing per-ordering CSVs named "
            "'sequential.csv', 'random.csv', 'adversarial.csv'."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write overall / per-cell / LaTeX outputs into.",
    )
    return parser.parse_args(argv)


def load_ordering_rows(path: Path, expected_ordering: str) -> list[Row]:
    if not path.exists():
        raise SystemExit(f"missing input CSV: {path}")
    out: list[Row] = []
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            ordering = raw.get("class_ordering") or expected_ordering
            if ordering != expected_ordering:
                raise SystemExit(
                    f"{path} contains class_ordering={ordering!r}; "
                    f"expected {expected_ordering!r}"
                )
            out.append(
                Row(
                    instance_id=raw["instance_id"],
                    cell=(
                        int(raw["N"]),
                        int(raw["M"]),
                        raw["correlation"],
                        float(raw["f"]),
                    ),
                    profit=(
                        float(raw["profit"]) if raw.get("profit") not in (None, "") else None
                    ),
                    status=raw["status"],
                    wall_time_s=float(raw["wall_time_s"]),
                    ordering=ordering,
                )
            )
    return out


def joined_table(
    per_ordering: dict[str, list[Row]],
) -> dict[str, dict[str, Row]]:
    """Return `{ordering: {instance_id: Row}}`, asserting matching sets."""
    by_ordering: dict[str, dict[str, Row]] = {
        ordering: {row.instance_id: row for row in rows}
        for ordering, rows in per_ordering.items()
    }
    reference = set(by_ordering[ORDERINGS[0]])
    for ordering in ORDERINGS[1:]:
        if set(by_ordering[ordering]) != reference:
            missing = reference - set(by_ordering[ordering])
            extra = set(by_ordering[ordering]) - reference
            raise SystemExit(
                "instance set mismatch between orderings: "
                f"{ordering!r} missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}"
            )
    return by_ordering


def per_instance_gaps(
    joined: dict[str, dict[str, Row]],
) -> dict[str, dict[str, float | None]]:
    """For each instance, compute paired gap_pct[ordering] vs best-of-three."""
    instance_ids = sorted(joined[ORDERINGS[0]])
    out: dict[str, dict[str, float | None]] = {}
    for inst in instance_ids:
        profits = {ord: joined[ord][inst].profit for ord in ORDERINGS}
        valid = [p for p in profits.values() if p is not None]
        if not valid or max(valid) <= 0:
            out[inst] = {ord: None for ord in ORDERINGS}
            continue
        best = max(valid)
        out[inst] = {
            ord: (
                None
                if profits[ord] is None
                else (best - profits[ord]) / best * 100.0
            )
            for ord in ORDERINGS
        }
    return out


def per_instance_wins(
    joined: dict[str, dict[str, Row]],
) -> dict[str, dict[str, bool]]:
    """A "win" means tied with `max(profit)` across the three orderings."""
    instance_ids = sorted(joined[ORDERINGS[0]])
    out: dict[str, dict[str, bool]] = {}
    for inst in instance_ids:
        profits = {ord: joined[ord][inst].profit for ord in ORDERINGS}
        valid = [p for p in profits.values() if p is not None]
        if not valid:
            out[inst] = {ord: False for ord in ORDERINGS}
            continue
        best = max(valid)
        out[inst] = {
            ord: (profits[ord] is not None and math.isclose(profits[ord], best))
            for ord in ORDERINGS
        }
    return out


def overall_stats(
    joined: dict[str, dict[str, Row]],
    gaps: dict[str, dict[str, float | None]],
    wins: dict[str, dict[str, bool]],
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for ordering in ORDERINGS:
        ord_gaps = [g[ordering] for g in gaps.values() if g[ordering] is not None]
        ord_walls = [
            row.wall_time_s for row in joined[ordering].values()
        ]
        ord_timeouts = sum(
            1 for row in joined[ordering].values() if row.status == "timeout"
        )
        ord_errors = sum(
            1 for row in joined[ordering].values() if row.status == "error"
        )
        ord_wins = sum(1 for w in wins.values() if w[ordering])
        n = len(joined[ordering])
        out[ordering] = {
            "n_instances": n,
            "mean_gap_pct": statistics.fmean(ord_gaps) if ord_gaps else 0.0,
            "median_gap_pct": statistics.median(ord_gaps) if ord_gaps else 0.0,
            "std_gap_pct": statistics.pstdev(ord_gaps) if len(ord_gaps) > 1 else 0.0,
            "max_gap_pct": max(ord_gaps) if ord_gaps else 0.0,
            "win_rate": ord_wins / n if n else 0.0,
            "win_count": ord_wins,
            "timeout_count": ord_timeouts,
            "error_count": ord_errors,
            "mean_wall_s": statistics.fmean(ord_walls) if ord_walls else 0.0,
        }
    return out


def per_cell_stats(
    joined: dict[str, dict[str, Row]],
    gaps: dict[str, dict[str, float | None]],
    wins: dict[str, dict[str, bool]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    cells: dict[tuple[int, int, str, float], list[str]] = defaultdict(list)
    for inst, row in joined[ORDERINGS[0]].items():
        cells[row.cell].append(inst)
    for cell in sorted(cells):
        n, m, correlation, f = cell
        for ordering in ORDERINGS:
            insts = cells[cell]
            ord_gaps = [
                gaps[i][ordering] for i in insts if gaps[i][ordering] is not None
            ]
            ord_timeouts = sum(
                1 for i in insts if joined[ordering][i].status == "timeout"
            )
            ord_wins = sum(1 for i in insts if wins[i][ordering])
            out.append(
                {
                    "N": n,
                    "M": m,
                    "correlation": correlation,
                    "f": f,
                    "class_ordering": ordering,
                    "n_instances": len(insts),
                    "mean_gap_pct": (
                        statistics.fmean(ord_gaps) if ord_gaps else 0.0
                    ),
                    "median_gap_pct": (
                        statistics.median(ord_gaps) if ord_gaps else 0.0
                    ),
                    "max_gap_pct": max(ord_gaps) if ord_gaps else 0.0,
                    "win_count": ord_wins,
                    "timeout_count": ord_timeouts,
                }
            )
    return out


def per_instance_long(
    joined: dict[str, dict[str, Row]],
    gaps: dict[str, dict[str, float | None]],
) -> Iterable[dict[str, object]]:
    instance_ids = sorted(joined[ORDERINGS[0]])
    for inst in instance_ids:
        for ordering in ORDERINGS:
            row = joined[ordering][inst]
            yield {
                "instance_id": inst,
                "N": row.cell[0],
                "M": row.cell[1],
                "correlation": row.cell[2],
                "f": row.cell[3],
                "class_ordering": ordering,
                "status": row.status,
                "profit": "" if row.profit is None else int(row.profit),
                "wall_time_s": row.wall_time_s,
                "gap_pct_vs_best": (
                    "" if gaps[inst][ordering] is None else gaps[inst][ordering]
                ),
            }


def _format_pct(value: float) -> str:
    """Project cadence: |x| >= 1 -> 2 decimals; |x| < 1 -> 4 decimals."""
    if abs(value) >= 1.0:
        return f"{value:.2f}"
    return f"{value:.4f}"


def write_latex_table(path: Path, overall: dict[str, dict[str, float | int]]) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "ordering & N & wins & mean gap (\\%) & median gap (\\%) & "
        "std gap (\\%) & timeouts \\\\",
        "\\hline",
    ]
    for ordering in ORDERINGS:
        block = overall[ordering]
        lines.append(
            f"{ordering} & {block['n_instances']} & {block['win_count']} & "
            f"{_format_pct(float(block['mean_gap_pct']))} & "
            f"{_format_pct(float(block['median_gap_pct']))} & "
            f"{_format_pct(float(block['std_gap_pct']))} & "
            f"{block['timeout_count']} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def write_per_cell_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_per_instance_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_ordering = {
        ordering: load_ordering_rows(
            args.results_dir / f"{ordering}.csv", ordering
        )
        for ordering in ORDERINGS
    }
    joined = joined_table(per_ordering)
    gaps = per_instance_gaps(joined)
    wins = per_instance_wins(joined)
    overall = overall_stats(joined, gaps, wins)
    per_cell = per_cell_stats(joined, gaps, wins)

    overall_path = args.out_dir / "class_ordering_overall.json"
    overall_path.write_text(json.dumps(overall, indent=2, sort_keys=True) + "\n")

    write_per_cell_csv(args.out_dir / "class_ordering_per_cell.csv", per_cell)
    write_per_instance_csv(
        args.out_dir / "class_ordering_per_instance.csv",
        per_instance_long(joined, gaps),
    )
    write_latex_table(args.out_dir / "class_ordering_summary.tex", overall)

    print(f"results_dir: {args.results_dir}")
    print(f"out_dir: {args.out_dir}")
    print(f"n_instances: {overall[ORDERINGS[0]]['n_instances']}")
    for ordering in ORDERINGS:
        block = overall[ordering]
        print(
            f"{ordering:>12s}: mean_gap={_format_pct(float(block['mean_gap_pct']))}% "
            f"median={_format_pct(float(block['median_gap_pct']))}% "
            f"std={_format_pct(float(block['std_gap_pct']))}% "
            f"wins={block['win_count']}/{block['n_instances']} "
            f"timeouts={block['timeout_count']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
