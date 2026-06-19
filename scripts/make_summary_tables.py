r"""Emit per-N paired-comparison summary tables for §3.9 of the manuscript.

Reads ``comparison_summary/paired_profit_gaps.csv`` from a pinned
final-experiments archive and writes two LaTeX `tabular` fragments,
one per requested baseline:

* ``hld_vs_partition_summary.tex`` — paired comparison of HLD against
  the Partition-Optimal naive-decomposition baseline at each scale.
* ``hld_vs_highs_summary.tex`` — paired comparison of HLD against
  the HiGHS open-source mixed-integer reference (only available at
  N ≤ 10\,000; HiGHS does not return an incumbent at N = 100\,000).

Both fragments are designed to be ``\input{}``-ed inside an enclosing
``table`` float and to resolve the cross-references
``tab:hld_vs_partition_summary`` / ``tab:hld_vs_highs_summary`` quoted
in §3.9 of ``knapsack-optimization-paper/main.tex``.

Numbers reproduce the §3.9 prose exactly: HLD vs PO at N=100k gives
a paired median gain of +17.19% (1\,172 / 2\,046 wins); HLD vs HiGHS
at N=10k gives a paired median gap of -0.003%.

Usage:

    uv run python scripts/make_summary_tables.py \
        --paired-csv path/to/comparison_summary/paired_profit_gaps.csv \
        --out-dir results/final_experiments/paper_tables
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

BASELINES = {
    "partition_optimal": {
        "label_short": "Partition-Optimal",
        "label_long": "the Partition-Optimal naive-decomposition baseline",
        "fname": "hld_vs_partition_summary.tex",
        "expected_Ns": (1_000, 10_000, 100_000),
    },
    "highs": {
        "label_short": "HiGHS",
        "label_long": "the HiGHS open-source mixed-integer reference",
        "fname": "hld_vs_highs_summary.tex",
        "expected_Ns": (1_000, 10_000),
    },
}

# Combined multi-baseline summary (Opus round-2 M1 / AE-2a). Reports HLD's
# paired profit gain against every baseline at each scale, median-focused with
# a 5%-winsorized mean (S3) so the long Partition-Optimal right tail does not
# dominate. Order: greedy floor, then published heuristics, then the naive
# decomposition baseline, then the exact open-source reference.
COMBINED_FNAME = "hld_vs_baselines_summary.tex"
COMBINED_ORDER: tuple[tuple[str, str], ...] = (
    ("greedy_max_ratio", "Greedy-MaxRatio"),
    ("trs2008", "TRS-2008"),
    ("bissa", "BISSA"),
    ("partition_optimal", "Partition-Optimal"),
    ("highs", "HiGHS"),
)
WINSOR_LIMIT = 0.05

# 300 s full-grid HLD vs open-source heuristics (Opus round-2 §8.3 / M1).
# Heuristic profits are time-budget-independent, so we reuse the per-instance
# baseline profits already in paired_profit_gaps.csv and only swap HLD's 60 s
# profit for its 300 s profit (per-instance, from the time-limit sensitivity
# run), recomputing each paired gain. Only the largest scale is in scope: the
# reviewer's concern is N = 100\,000 under an extended budget.
COMBINED_300S_FNAME = "hld_vs_baselines_300s.tex"
HEURISTIC_300S_ORDER: tuple[tuple[str, str], ...] = (
    ("greedy_max_ratio", "Greedy-MaxRatio"),
    ("trs2008", "TRS-2008"),
    ("bissa", "BISSA"),
)
N_300S = 100_000
TIME_LIMIT_300S = "300"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--hld-300s-csv",
        type=Path,
        default=None,
        help=(
            "Optional per-instance time-limit-sensitivity CSV "
            "(run_time_limit_sensitivity.py output). When given, also emits "
            f"{COMBINED_300S_FNAME}: HLD@300s vs each open-source heuristic at "
            f"N={N_300S:,}."
        ),
    )
    parser.add_argument(
        "--time-limit-s",
        default=TIME_LIMIT_300S,
        help="time_limit_s value to select from --hld-300s-csv (default: 300).",
    )
    return parser.parse_args()


def _aggregate(rows: list[dict[str, str]]) -> list[dict[str, float | int]]:
    """Compute paired aggregates per N for the input subset."""
    by_N: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_N[int(row["N"])].append(float(row["hld_vs_baseline_gain_pct"]))
    out: list[dict[str, float | int]] = []
    for N in sorted(by_N):
        gains = by_N[N]
        out.append(
            {
                "N": N,
                "n": len(gains),
                "mean": statistics.mean(gains),
                "winsor_mean": _winsorized_mean(gains),
                "median": statistics.median(gains),
                "wins": sum(1 for g in gains if g > 0),
                "losses": sum(1 for g in gains if g < 0),
                "ties": sum(1 for g in gains if g == 0),
            }
        )
    return out


def _format_pct(value: float) -> str:
    """Match the §3.9 prose cadence: 2 decimals for |x| ≥ 1; extra precision below."""
    if abs(value) >= 1.0:
        return f"{value:+.2f}"
    if abs(value) >= 0.01:
        return f"{value:+.2f}"
    return f"{value:+.3f}"


def _winsorized_mean(values: list[float], limit: float = WINSOR_LIMIT) -> float:
    """Mean after clamping each tail to its `limit` quantile (S3 robustness).

    Unlike a trimmed mean this keeps all observations but caps extreme tails,
    so the explosive Partition-Optimal right tail at N=100k stops dominating
    while the central tendency stays comparable across baselines.
    """
    xs = sorted(values)
    n = len(xs)
    k = int(n * limit)
    if k == 0 or n - 2 * k <= 0:
        return statistics.mean(xs)
    lo, hi = xs[k], xs[n - k - 1]
    return statistics.mean(min(max(x, lo), hi) for x in xs)


def _format_int(value: int) -> str:
    """Format an integer with LaTeX thin-space thousands separator."""
    return f"{value:,}".replace(",", r"\,")


def _emit_table(
    aggregates: list[dict[str, float | int]],
    label_long: str,
) -> str:
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\hline",
        r"$N$ & paired $n$ & mean gain (\%) & winsor.\ mean (\%) & "
        + r"median gain (\%) & HLD wins & ref wins & ties \\",
        r"\hline",
    ]
    for row in aggregates:
        N = int(row["N"])
        cells = [
            f"${_format_int(N)}$",
            _format_int(int(row["n"])),
            _format_pct(float(row["mean"])),
            _format_pct(float(row["winsor_mean"])),
            _format_pct(float(row["median"])),
            _format_int(int(row["wins"])),
            _format_int(int(row["losses"])),
            _format_int(int(row["ties"])),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def _emit_combined_table(
    rows: list[dict[str, str]],
    order: tuple[tuple[str, str], ...] = COMBINED_ORDER,
    header_comment: tuple[str, ...] = (
        r"% Paired HLD vs each baseline, aggregated by N (median-focused; mean",
        r"% is 5\%-winsorized per reviewer S3 to tame the long Partition-Optimal tail).",
        r"% Positive gain = HLD better. Generated by scripts/make_summary_tables.py.",
    ),
) -> str:
    """One median-focused HLD-vs-baselines table grouped by algorithm."""
    header = [
        *header_comment,
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Baseline & $N$ & paired $n$ & median gain (\%) & winsor.\ mean (\%) & "
        + r"HLD wins & baseline wins & ties \\",
        r"\hline",
    ]
    body: list[str] = []
    for baseline, label in order:
        subset = [r for r in rows if r["baseline_solver"] == baseline]
        if not subset:
            print(f"warning: no rows for baseline={baseline!r}; skipping in combined table")
            continue
        agg = _aggregate(subset)
        gains_by_n = {
            int(r["N"]): [
                float(x["hld_vs_baseline_gain_pct"])
                for x in subset
                if int(x["N"]) == int(r["N"])
            ]
            for r in agg
        }
        for i, row in enumerate(agg):
            N = int(row["N"])
            cells = [
                label if i == 0 else "",
                f"${_format_int(N)}$",
                _format_int(int(row["n"])),
                _format_pct(float(row["median"])),
                _format_pct(_winsorized_mean(gains_by_n[N])),
                _format_int(int(row["wins"])),
                _format_int(int(row["losses"])),
                _format_int(int(row["ties"])),
            ]
            body.append(" & ".join(cells) + r" \\")
        body.append(r"\hline")
    return "\n".join(header + body + [r"\end{tabular}", ""])


def load_hld_300s_profits(
    csv_path: Path, time_limit_s: str = TIME_LIMIT_300S
) -> dict[str, tuple[str, float]]:
    """Map instance_id -> (status, profit) for HLD at the requested budget.

    Reads the per-instance time-limit-sensitivity CSV and keeps only HLD rows
    at the selected ``time_limit_s``. Profit is parsed leniently because an
    infeasible (timed-out) row may carry an empty profit field.
    """
    out: dict[str, tuple[str, float]] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("solver") != "hld":
                continue
            if str(row.get("time_limit_s")) != str(time_limit_s):
                continue
            try:
                profit = float(row["profit"]) if row.get("profit") else 0.0
            except ValueError:
                profit = 0.0
            out[row["instance_id"]] = (row.get("status", ""), profit)
    return out


def build_300s_rows(
    paired_rows: list[dict[str, str]],
    hld_300s: dict[str, tuple[str, float]],
    *,
    n_filter: int = N_300S,
    order: tuple[tuple[str, str], ...] = HEURISTIC_300S_ORDER,
) -> tuple[list[dict[str, str]], int]:
    """Re-pair HLD@300s profit against each heuristic's per-instance profit.

    Returns synthetic paired rows in the same schema as
    ``paired_profit_gaps.csv`` (so the existing aggregation/emitter work
    unchanged) plus the count of paired rows whose HLD@300s profit was missing
    from ``hld_300s`` (unmatched), which the caller can warn on.
    """
    wanted = {b for b, _ in order}
    synth: list[dict[str, str]] = []
    unmatched = 0
    for row in paired_rows:
        if row["baseline_solver"] not in wanted or int(row["N"]) != n_filter:
            continue
        hit = hld_300s.get(row["instance_id"])
        if hit is None:
            unmatched += 1
            continue
        hld_status, hld_profit = hit
        baseline_profit = float(row["baseline_profit"])
        baseline_feasible = row["baseline_status"] == "feasible"
        hld_feasible = hld_status == "feasible" and hld_profit > 0
        if not baseline_feasible and not hld_feasible:
            continue  # both fail: no informative pairing
        if hld_feasible and baseline_feasible and baseline_profit != 0:
            gain = 100.0 * (hld_profit - baseline_profit) / baseline_profit
        elif hld_feasible and not baseline_feasible:
            gain = 100.0  # HLD recovers profit where the heuristic gives none
        else:
            gain = -100.0  # HLD times out where the heuristic is feasible
        synth.append(
            {
                **row,
                "hld_status": hld_status,
                "hld_profit": str(hld_profit),
                "hld_vs_baseline_gain_pct": str(gain),
                "winner": "hld" if gain > 0 else "baseline" if gain < 0 else "tie",
            }
        )
    return synth, unmatched


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.paired_csv.open() as fh:
        rows = list(csv.DictReader(fh))

    for baseline, spec in BASELINES.items():
        subset = [r for r in rows if r["baseline_solver"] == baseline]
        if not subset:
            raise SystemExit(f"no rows for baseline={baseline!r} in {args.paired_csv}")
        agg = _aggregate(subset)
        seen_Ns = tuple(row["N"] for row in agg)
        if seen_Ns != spec["expected_Ns"]:
            print(f"warning: baseline={baseline} Ns {seen_Ns} != expected {spec['expected_Ns']}")
        out_path = args.out_dir / spec["fname"]
        out_path.write_text(_emit_table(agg, spec["label_long"]))
        n_total = sum(int(r["n"]) for r in agg)
        print(
            f"{baseline:20s} -> {out_path.name:40s} "
            f"({len(agg)} N-rows, {n_total:,} paired instances)"
        )

    combined_path = args.out_dir / COMBINED_FNAME
    combined_path.write_text(_emit_combined_table(rows))
    print(f"{'combined':20s} -> {combined_path.name:40s} ({len(COMBINED_ORDER)} baselines)")

    if args.hld_300s_csv is not None:
        hld_300s = load_hld_300s_profits(args.hld_300s_csv, args.time_limit_s)
        synth, unmatched = build_300s_rows(rows, hld_300s)
        if not synth:
            raise SystemExit(
                f"no HLD@{args.time_limit_s}s pairings built from "
                f"{args.hld_300s_csv} (expected per-instance HLD rows at "
                f"N={N_300S:,}); did the full-grid run finish?"
            )
        if unmatched:
            print(
                f"warning: {unmatched} N={N_300S:,} paired rows had no "
                f"HLD@{args.time_limit_s}s profit; excluded from 300s table"
            )
        comment = (
            r"% HLD@300s vs open-source heuristics at N=100\,000 (Opus round-2 §8.3).",
            r"% Heuristic profits are budget-independent (reused from the 60s grid);",
            r"% only HLD's profit is re-measured at the 300s budget. Positive gain =",
            r"% HLD better. Generated by scripts/make_summary_tables.py --hld-300s-csv.",
        )
        path_300s = args.out_dir / COMBINED_300S_FNAME
        path_300s.write_text(
            _emit_combined_table(synth, order=HEURISTIC_300S_ORDER, header_comment=comment)
        )
        print(
            f"{'combined@300s':20s} -> {path_300s.name:40s} "
            f"({len(HEURISTIC_300S_ORDER)} heuristics, {len(synth):,} pairings)"
        )


if __name__ == "__main__":
    main()
