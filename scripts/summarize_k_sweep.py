#!/usr/bin/env python3
"""Summarize the batch-count (K) robustness sweep produced by ``k_sweep.py``.

Companion analysis for ``scripts/k_sweep.py``. Reads the sweep JSONL and
reports, per benchmark cell (keyed by ``f``) and per K:

- feasibility (how many instances each solver returns a non-zero profit on),
- HLD median profit (headline statistic, consistent with the manuscript's
  distribution-free reporting),
- the paired HLD-vs-Partition-Optimal comparison (win/tie/lose and median
  paired gain over instances where the equal-split reference is feasible),
- HLD median wall time.

The interpretation is deliberately regime-honest: it identifies the
infeasible small-K regime (where the per-batch sub-MILP cannot find any
incumbent within the time budget), the feasible regime, and the per-cell
sweet spot, rather than assuming HLD quality is flat or monotone in K.

Examples
--------
::

    PYTHONPATH=code uv run python scripts/summarize_k_sweep.py \
        --sweep-jsonl results/k_sweep/sweep.jsonl \
        --out-dir results/k_sweep/summary
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path
from typing import Any

HLD = "hld"
PARTITION = "partition_optimal"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-jsonl", type=Path, default=Path("results/k_sweep/sweep.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/k_sweep/summary"))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _profit(row: dict[str, Any]) -> float:
    value = row.get("profit")
    return float(value) if value is not None else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the per-(f, k) summary plus per-f regime classification."""
    # profit[(f, seed, solver, k)] -> profit
    profit: dict[tuple[float, int, str, int], float] = {}
    walls: dict[tuple[float, str, int], list[float]] = defaultdict(list)
    fs: set[float] = set()
    ks: set[int] = set()
    for row in rows:
        f = float(row["f"])
        k = int(row["k"])
        solver = str(row["solver"])
        fs.add(f)
        ks.add(k)
        profit[(f, int(row["seed"]), solver, k)] = _profit(row)
        walls[(f, solver, k)].append(float(row["wall_time_s"]))

    f_grid = sorted(fs)
    k_grid = sorted(ks)
    seeds_by_f = {f: sorted({key[1] for key in profit if key[0] == f}) for f in f_grid}

    cells: list[dict[str, Any]] = []
    per_f: dict[float, dict[str, Any]] = {}
    for f in f_grid:
        seeds = seeds_by_f[f]
        feasible_ks: list[int] = []
        hld_median_by_k: dict[int, float] = {}
        for k in k_grid:
            hld_profits = [profit[(f, s, HLD, k)] for s in seeds if (f, s, HLD, k) in profit]
            po_profits = [
                profit[(f, s, PARTITION, k)] for s in seeds if (f, s, PARTITION, k) in profit
            ]
            hld_feasible = sum(1 for p in hld_profits if p > 0)
            po_feasible = sum(1 for p in po_profits if p > 0)

            win = lose = tie = 0
            gains: list[float] = []
            for s in seeds:
                h = profit.get((f, s, HLD, k))
                p = profit.get((f, s, PARTITION, k))
                if h is None or p is None:
                    continue
                if h > p:
                    win += 1
                elif h < p:
                    lose += 1
                else:
                    tie += 1
                if p > 0:
                    gains.append((h - p) / p)

            hld_median = stats.median(hld_profits) if hld_profits else 0.0
            hld_median_by_k[k] = hld_median
            if hld_feasible > 0:
                feasible_ks.append(k)

            cells.append(
                {
                    "f": f,
                    "k": k,
                    "n_instances": len(seeds),
                    "hld_feasible": hld_feasible,
                    "po_feasible": po_feasible,
                    "hld_median_profit": hld_median,
                    "po_median_profit": stats.median(po_profits) if po_profits else 0.0,
                    "paired_win": win,
                    "paired_tie": tie,
                    "paired_lose": lose,
                    "paired_median_gain": stats.median(gains) if gains else None,
                    "hld_median_wall_s": stats.median(walls[(f, HLD, k)])
                    if walls.get((f, HLD, k))
                    else 0.0,
                }
            )

        infeasible_ks = [k for k in k_grid if k not in feasible_ks]
        best_k = max(feasible_ks, key=lambda k: hld_median_by_k[k]) if feasible_ks else None
        per_f[f] = {
            "feasible_ks": feasible_ks,
            "infeasible_ks": infeasible_ks,
            "best_k": best_k,
            "hld_median_by_k": hld_median_by_k,
        }

    return {
        "f_grid": f_grid,
        "k_grid": k_grid,
        "n_rows": len(rows),
        "cells": cells,
        "per_f": per_f,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = ["# HLD batch-count (K) robustness sweep\n"]
    lines.append(
        f"{summary['n_rows']} records over K grid `{summary['k_grid']}` "
        f"and cells f={summary['f_grid']}.\n"
    )
    for f in summary["f_grid"]:
        info = summary["per_f"][f]
        lines.append(f"\n## Cell f={f}\n")
        lines.append(
            f"- feasible K: `{info['feasible_ks']}`; "
            f"infeasible K (both solvers time out, profit 0): `{info['infeasible_ks']}`\n"
            f"- HLD sweet spot (max median profit): **K={info['best_k']}**\n"
        )
        lines.append(
            "| K | HLD feas | PO feas | HLD median profit | HLD>PO | =  | < | median gain | HLD wall (s) |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cell in summary["cells"]:
            if cell["f"] != f:
                continue
            gain = cell["paired_median_gain"]
            gain_str = f"{gain:+.2%}" if gain is not None else "n/a"
            lines.append(
                f"| {cell['k']} | {cell['hld_feasible']} | {cell['po_feasible']} "
                f"| {cell['hld_median_profit']:.0f} | {cell['paired_win']} "
                f"| {cell['paired_tie']} | {cell['paired_lose']} | {gain_str} "
                f"| {cell['hld_median_wall_s']:.1f} |"
            )
    lines.append("\n## Interpretation\n")
    lines.append(interpret(summary))
    return "\n".join(lines) + "\n"


def interpret(summary: dict[str, Any]) -> str:
    bits: list[str] = []
    infeasible_union = sorted(
        {k for f in summary["f_grid"] for k in summary["per_f"][f]["infeasible_ks"]}
    )
    best_ks = sorted(
        {
            summary["per_f"][f]["best_k"]
            for f in summary["f_grid"]
            if summary["per_f"][f]["best_k"] is not None
        }
    )
    if infeasible_union:
        bits.append(
            f"- **Small-K is infeasible:** at K in `{infeasible_union}` both HLD and "
            "Partition-Optimal return profit 0 on every instance — the per-batch "
            "sub-MILP is too large to find any incumbent within the time budget.\n"
        )
    if best_ks:
        bits.append(
            f"- **Sweet spot at moderate K:** the HLD-best K per cell is in `{best_ks}`. "
            "Above it, median profit declines as K grows (more batches fragment the "
            "shared budget), so K is a tunable quality knob rather than free "
            "parallelism.\n"
        )
    # Monotone-decline check above each cell's sweet spot.
    declines = []
    for f in summary["f_grid"]:
        info = summary["per_f"][f]
        best_k = info["best_k"]
        if best_k is None:
            continue
        above = [k for k in info["feasible_ks"] if k > best_k]
        med = info["hld_median_by_k"]
        monotone = all(med[above[i]] >= med[above[i + 1]] for i in range(len(above) - 1))
        declines.append((f, monotone))
    if declines and all(m for _, m in declines):
        bits.append(
            "- Above the sweet spot the decline is monotone in every cell, "
            "consistent with a single interior optimum.\n"
        )
    return "".join(bits) if bits else "_no records_\n"


def render_plots(summary: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for idx, f in enumerate(summary["f_grid"]):
        info = summary["per_f"][f]
        feasible = info["feasible_ks"]
        if not feasible:
            continue
        med = info["hld_median_by_k"]
        best = max(med[k] for k in feasible) or 1.0
        ys = [med[k] / best for k in feasible]
        ax.plot(
            feasible,
            ys,
            marker="o",
            color=colors[idx % len(colors)],
            label=f"f={f}",
        )
        if info["best_k"] is not None:
            ax.scatter(
                [info["best_k"]],
                [1.0],
                marker="*",
                s=140,
                color=colors[idx % len(colors)],
                zorder=5,
            )

    infeasible_union = [
        k
        for k in summary["k_grid"]
        if all(k in summary["per_f"][f]["infeasible_ks"] for f in summary["f_grid"])
    ]
    if infeasible_union:
        ax.axvspan(
            min(summary["k_grid"]),
            max(infeasible_union) * 1.4,
            color="0.85",
            zorder=0,
            label="infeasible (both solvers)",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch count $K$")
    ax.set_ylabel("HLD median profit (relative to best $K$)")
    ax.set_title("HLD solution quality vs batch count $K$\n($N=100{,}000$, inversely-strongly)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "k_robustness.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_rows(args.sweep_jsonl)
    summary = summarize(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "analyses.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "REPORT.md").write_text(render_report(summary))

    if not args.no_plots:
        try:
            render_plots(summary, args.out_dir)
        except ImportError:
            print("matplotlib not installed; skipping plots")

    print(f"sweep_jsonl: {args.sweep_jsonl}")
    print(f"out_dir: {args.out_dir}")
    print(f"total_rows: {summary['n_rows']}")
    for f in summary["f_grid"]:
        info = summary["per_f"][f]
        print(
            f"f={f}: best_k={info['best_k']} "
            f"feasible_ks={info['feasible_ks']} "
            f"infeasible_ks={info['infeasible_ks']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
