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
DEFAULT_OUT_DIR = Path("figures")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-csv",
        type=Path,
        default=DEFAULT_PAIRED_CSV,
        help=f"comparison_summary/paired_profit_gaps.csv (default: {DEFAULT_PAIRED_CSV}).",
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
        "--only",
        type=str,
        action="append",
        default=None,
        help="Generate only this named figure (repeatable). Default: all.",
    )
    return parser.parse_args(argv)


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


FIGURE_GENERATORS: dict[str, Callable[[argparse.Namespace], None]] = {
    "hld_vs_partition_paired_gains": make_hld_vs_partition_paired_gains,
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
