#!/usr/bin/env python3
"""Per-instance appendix tables (Task 3.6.1 of revision-finalization-2026).

Reads the pinned `comparison_summary/paired_profit_gaps.csv` and emits
one compact appendix table per scale (`N`), with median / worst-case
HLD-vs-reference gain (%) and median wall-times per (M, correlation,
f) cell. Reference solver:

- N=1000, N=10000 -> HiGHS (exact)
- N=100000        -> Partition-Optimal (naive decomposition baseline)

Outputs (in `--out-dir`):

- `appendix_N{n}.csv`  -- machine-readable per-cell table
- `appendix_N{n}.tex`  -- LaTeX longtable environment (page-breaking)

Number cadence follows the project-wide rule:
percentages with |x| >= 1 use 2 decimals; sub-1 % use 4 decimals.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Reference solver per N. Q3 of decisions/2026-05-25_open_questions.md.
REFERENCE_PER_N: dict[int, str] = {
    1000: "highs",
    10000: "highs",
    100000: "partition_optimal",
}

# Per-scale captions embedded in the emitted longtable (the appendix now lives in
# the supplementary material as page-breaking longtables). Kept here so a
# regeneration reproduces the exact caption text shipped in the manuscript.
CAPTIONS: dict[int, str] = {
    1000: (
        r"Per-cell paired profit gap of HLD vs HiGHS at $N = 1\,000$ (60 cells). "
        r"Negative \emph{med gain (\%)} means HiGHS attains a strictly higher profit "
        r"on the median seed; positive values mean HLD wins. HLD is at or within tenths "
        r"of a percent of HiGHS on the median of every cell at this scale."
    ),
    10000: (
        r"Per-cell paired profit gap of HLD vs HiGHS at $N = 10\,000$ (55 cells; five "
        r"cells where HiGHS returned no incumbent within the 60-second budget are omitted "
        r"because the paired comparison is undefined). HLD remains at parity with HiGHS on "
        r"the median of every cell except a handful of strongly- and "
        r"inversely-strongly-correlated low-$f$ cells where HiGHS finds a strictly better "
        r"incumbent under the same wall-clock budget; see "
        r"Section~\ref{sec:large_scale_validation} for the aggregate."
    ),
    100000: (
        r"Per-cell paired profit gap of HLD vs Partition-Optimal at $N = 100\,000$ "
        r"(60 cells). HiGHS is omitted because it does not return an incumbent within the "
        r"60-second budget at this scale. Positive \emph{med gain (\%)} means HLD recovers "
        r"profit that the equal-budget decomposition leaves on the table; negative values "
        r"mean Partition-Optimal wins on the median seed. The bimodal cell-level picture "
        r"noted in Section~\ref{sec:large_scale_validation} (HLD wins 33 of 60 cells; "
        r"Partition-Optimal wins the remaining 27, predominantly on strongly- and "
        r"inversely-strongly-correlated cells at low $f$) is visible directly in this table."
    ),
}

CSV_FIELDS = [
    "N",
    "M",
    "correlation",
    "f",
    "reference_solver",
    "n_paired",
    "hld_median_wall_s",
    "hld_max_wall_s",
    "reference_median_wall_s",
    "median_gain_pct",
    "worst_loss_pct",
    "worst_gain_pct",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-csv",
        type=Path,
        required=True,
        help="Path to comparison_summary/paired_profit_gaps.csv from the pinned archive.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write appendix_N{n}.{csv,tex} into.",
    )
    return parser.parse_args(argv)


def load_paired(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def aggregate_by_cell(
    rows: list[dict[str, str]],
) -> dict[int, list[dict[str, object]]]:
    """Group paired rows by N, then by cell, keeping only the canonical reference."""
    out: dict[int, list[dict[str, object]]] = {}
    by_n: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        n = int(row["N"])
        if n not in REFERENCE_PER_N:
            continue
        if row["baseline_solver"] != REFERENCE_PER_N[n]:
            continue
        by_n[n].append(row)

    for n, n_rows in by_n.items():
        cells: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in n_rows:
            key = (int(row["M"]), row["correlation"], row["f"])
            cells[key].append(row)

        per_cell: list[dict[str, object]] = []
        for (m, correlation, f), cell_rows in sorted(cells.items()):
            gains = [float(r["hld_vs_baseline_gain_pct"]) for r in cell_rows]
            hld_walls = [float(r["hld_wall_time_s"]) for r in cell_rows]
            ref_walls = [float(r["baseline_wall_time_s"]) for r in cell_rows]
            per_cell.append(
                {
                    "N": n,
                    "M": m,
                    "correlation": correlation,
                    "f": f,
                    "reference_solver": REFERENCE_PER_N[n],
                    "n_paired": len(cell_rows),
                    "hld_median_wall_s": statistics.median(hld_walls),
                    "hld_max_wall_s": max(hld_walls),
                    "reference_median_wall_s": statistics.median(ref_walls),
                    "median_gain_pct": statistics.median(gains),
                    "worst_loss_pct": min(gains),
                    "worst_gain_pct": max(gains),
                }
            )
        out[n] = per_cell
    return out


def _format_pct(value: float) -> str:
    """Project cadence: |x| >= 1 -> 2 decimals; |x| < 1 -> 4 decimals."""
    if abs(value) >= 1.0:
        return f"{value:+.2f}"
    return f"{value:+.4f}"


def _format_wall(value: float) -> str:
    if value >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def write_appendix_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _tex_escape_underscores(text: str) -> str:
    """Escape `_` for LaTeX text mode. Required for correlation /
    reference-solver labels that contain underscores (e.g. `inversely_strongly`,
    `partition_optimal`); without escaping these break compilation with
    "Missing $ inserted" because `_` is the subscript operator."""
    return text.replace("_", r"\_")


def write_appendix_tex(path: Path, n: int, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    ref = _tex_escape_underscores(str(rows[0]["reference_solver"]))
    label = f"tab:per_instance_N{n}"
    header = (
        "$M$ & corr. & $f$ & $n$ & "
        "HLD med (s) & HLD max (s) & ref med (s) & "
        "med gain (\\%) & worst gain (\\%) \\\\"
    )
    caption = CAPTIONS.get(n, f"Per-cell paired profit gap of HLD at $N = {n}$.")
    # longtable so the 55--60 row tables break across pages in the supplement.
    lines = [
        "\\begin{longtable}{rlrrrrrrr}",
        f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
        "\\hline",
        header,
        f"\\multicolumn{{9}}{{l}}{{reference: \\texttt{{{ref}}}}} \\\\",
        "\\hline",
        "\\endfirsthead",
        f"\\multicolumn{{9}}{{c}}{{\\footnotesize (Table~\\ref{{{label}}} continued)}} \\\\",
        "\\hline",
        header,
        "\\hline",
        "\\endhead",
    ]
    for row in rows:
        lines.append(
            f"{row['M']} & {_tex_escape_underscores(str(row['correlation']))} & {row['f']} & "
            f"{row['n_paired']} & "
            f"{_format_wall(float(row['hld_median_wall_s']))} & "
            f"{_format_wall(float(row['hld_max_wall_s']))} & "
            f"{_format_wall(float(row['reference_median_wall_s']))} & "
            f"{_format_pct(float(row['median_gain_pct']))} & "
            f"{_format_pct(float(row['worst_gain_pct']))} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{longtable}")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_paired(args.paired_csv)
    per_n = aggregate_by_cell(rows)
    for n in sorted(per_n):
        per_cell = per_n[n]
        if not per_cell:
            continue
        csv_path = args.out_dir / f"appendix_N{n}.csv"
        tex_path = args.out_dir / f"appendix_N{n}.tex"
        write_appendix_csv(csv_path, per_cell)
        write_appendix_tex(tex_path, n, per_cell)
        print(
            f"N={n}: {len(per_cell)} cells (reference={REFERENCE_PER_N[n]}) "
            f"-> {csv_path.name}, {tex_path.name}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
