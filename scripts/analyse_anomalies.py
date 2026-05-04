"""Anomaly investigation driver.

End-to-end:

1. Build (or reuse) the deterministic anomaly subset
   ``N=10 000, M=10, weakly correlated, f=0.5`` for a small fixed
   set of seeds.
2. Run HLD with ``N_iter`` swept over a grid (default 1..25), capture
   the full Phase-1 trajectory and Phase-3 batch wall times.
3. Use HiGHS to compute reference optimal profits.
4. Apply the H1 (degenerate dual basis) and H2 (sub-MILP straggler)
   tests to every record, write per-instance plots, and emit
   ``results/anomalies/REPORT.md``.

Examples
--------

Smoke run (small grid, one seed)::

    PYTHONPATH=code uv run python scripts/analyse_anomalies.py \
        --seeds 0 --n-iter-grid 1,5,10,20 --reference-time-limit-s 60

Full spec run (3 seeds, N_iter 1..25)::

    PYTHONPATH=code uv run python scripts/analyse_anomalies.py \
        --reference-time-limit-s 600

Re-analyse an existing sweep without re-running HLD::

    PYTHONPATH=code uv run python scripts/analyse_anomalies.py \
        --skip-sweep --sweep-path results/anomalies/sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from anomalies.analyse import (
    H1_LAMBDA_REL_SPREAD_THRESHOLD,
    H1_SIGN_FLIP_THRESHOLD,
    H2_MAX_BATCH_SHARE_THRESHOLD,
    analyse_record,
)
from anomalies.sweep import (
    DEFAULT_CELL,
    DEFAULT_N_ITER_GRID,
    DEFAULT_SEEDS,
    SweepRecord,
    ensure_anomaly_subset,
    load_sweep,
    run_sweep,
)

LOG = logging.getLogger("analyse_anomalies")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-root", type=Path, default=Path("instances"))
    p.add_argument("--out-dir", type=Path, default=Path("results/anomalies"))
    p.add_argument(
        "--sweep-path",
        type=Path,
        default=None,
        help="Where to write/read the JSONL sweep (default: <out-dir>/sweep.jsonl).",
    )
    p.add_argument(
        "--seeds",
        type=_parse_int_list,
        default=DEFAULT_SEEDS,
        help=f"Comma-separated seeds (default: {','.join(map(str, DEFAULT_SEEDS))}).",
    )
    p.add_argument(
        "--n-iter-grid",
        type=_parse_int_list,
        default=DEFAULT_N_ITER_GRID,
        help="Comma-separated N_iter values to sweep (default: 1..25).",
    )
    p.add_argument(
        "--reference-time-limit-s",
        type=float,
        default=600.0,
        help="HiGHS time limit per instance (default: 600 s).",
    )
    p.add_argument(
        "--eval-time-limit-s",
        type=float,
        default=None,
        help="Optional per-HLD-call time limit (default: none).",
    )
    p.add_argument("--sub-solver", type=str, default="highs")
    p.add_argument(
        "--reference-mip-rel-gap",
        type=float,
        default=None,
        help=(
            "Tighten HiGHS's `mip_rel_gap` for the reference solver. "
            "Use 1e-9 to remove the default-tolerance artefact (§4.3.5). "
            "Leave unset to keep HiGHS's default tolerance."
        ),
    )
    p.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the sweep and re-analyse an existing JSONL file.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plots (useful in headless / quick runs).",
    )
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x.strip())


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = args.sweep_path or (args.out_dir / "sweep.jsonl")

    if not args.skip_sweep:
        items = ensure_anomaly_subset(
            archive_root=args.archive_root,
            cell=DEFAULT_CELL,
            seeds=tuple(args.seeds),
        )
        LOG.info("anomaly subset: %d instances", len(items))
        run_sweep(
            items=items,
            n_iter_grid=tuple(args.n_iter_grid),
            sub_solver=args.sub_solver,
            reference_time_limit_s=args.reference_time_limit_s,
            reference_mip_rel_gap=args.reference_mip_rel_gap,
            eval_time_limit_s=args.eval_time_limit_s,
            out_path=sweep_path,
        )
        LOG.info("wrote sweep -> %s", sweep_path)

    raw = load_sweep(sweep_path)
    if not raw:
        LOG.error("sweep file %s is empty", sweep_path)
        return 1

    analyses = []
    for rec in raw:
        v = analyse_record(
            phase1_trajectory=rec["solver_metadata"].get("phase1_trajectory", []),
            phase3_batches=rec["solver_metadata"].get("phase3_batches", []),
            budget=rec["budget"],
        )
        analyses.append({"record": rec, "verdicts": v})

    analyses_path = args.out_dir / "analyses.json"
    analyses_path.write_text(
        json.dumps(
            [
                {
                    "inst_id": a["record"]["inst_id"],
                    "n_iter": a["record"]["n_iter"],
                    "optimality_gap": a["record"]["optimality_gap"],
                    "hld_wall_s": a["record"]["hld_wall_s"],
                    "verdicts": a["verdicts"].as_dict(),
                }
                for a in analyses
            ],
            indent=2,
        )
    )

    report = _render_report(analyses)
    (args.out_dir / "REPORT.md").write_text(report)
    LOG.info("wrote report -> %s", args.out_dir / "REPORT.md")

    if not args.no_plots:
        try:
            _render_plots(analyses, args.out_dir)
        except ImportError:
            LOG.warning("matplotlib not installed; skipping plots")

    return 0


def _render_report(analyses: list[dict]) -> str:
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for a in analyses:
        by_inst[a["record"]["inst_id"]].append(a)

    lines: list[str] = []
    lines.append("# HLD anomaly investigation (Phase D §4.3.2)\n")
    lines.append(
        "Mechanistic check of two hypotheses for the figure-anomaly behaviour "
        "(R2-M7, R2-A5):\n\n"
        "- **H1 — degenerate dual basis.** Phase-1 fails to converge: "
        f"`lambda_rel_spread > {H1_LAMBDA_REL_SPREAD_THRESHOLD:.0%}` over the "
        f"final iterations *or* ≥ {H1_SIGN_FLIP_THRESHOLD} sign flips of "
        "`total_cost - B`.\n"
        "- **H2 — sub-MILP straggler.** Slowest Phase-3 batch consumes "
        f"≥ {H2_MAX_BATCH_SHARE_THRESHOLD:.0%} of total Phase-3 wall time.\n"
    )

    n_records = len(analyses)
    h1_rate = sum(1 for a in analyses if a["verdicts"].h1_degenerate_dual) / n_records
    h2_rate = sum(1 for a in analyses if a["verdicts"].h2_straggler) / n_records
    h1_rate_low, h1_rate_high = _h1_rate_by_regime(analyses)
    mean_gap = mean(a["record"]["optimality_gap"] for a in analyses)
    mean_wall = mean(a["record"]["hld_wall_s"] for a in analyses)
    lines.append("## Summary\n")
    lines.append(
        f"- Records analysed: **{n_records}** "
        f"({len(by_inst)} instance(s) x {n_records // max(len(by_inst), 1)} N_iter values)\n"
        f"- Mean optimality gap: **{mean_gap:.2%}**\n"
        f"- Mean HLD wall time: **{mean_wall:.2f} s**\n"
        f"- H1 (degenerate dual) flagged: **{h1_rate:.0%}** overall "
        f"(low N_iter ≤ {H1_LOW_NITER_BOUNDARY}: **{h1_rate_low:.0%}**, "
        f"high N_iter > {H1_LOW_NITER_BOUNDARY}: **{h1_rate_high:.0%}**)\n"
        f"- H2 (sub-MILP straggler) flagged: **{h2_rate:.0%}** of records\n"
    )

    ref_lines = _reference_solver_summary(by_inst)
    if ref_lines:
        lines.append("\n### Reference solver status\n")
        lines.extend(ref_lines)

    lines.append("\n## Per-instance trajectories\n")
    for inst_id, recs in sorted(by_inst.items()):
        recs.sort(key=lambda r: r["record"]["n_iter"])
        lines.append(f"\n### `{inst_id}`\n")
        lines.append(
            "| N_iter | gap | HLD wall (s) | λ_final | gap to B | λ-spread | sign-flips | max-batch share | H1 | H2 |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|")
        for a in recs:
            r = a["record"]
            v = a["verdicts"]
            p1 = v.phase1
            p3 = v.phase3
            lines.append(
                f"| {r['n_iter']} "
                f"| {r['optimality_gap']:.2%} "
                f"| {r['hld_wall_s']:.2f} "
                f"| {p1.final_lambda:.3f} "
                f"| {p1.final_gap_to_budget:+d} "
                f"| {p1.lambda_rel_spread:.3f} "
                f"| {p1.sign_flips_in_window} "
                f"| {p3.max_batch_share:.0%} "
                f"| {'✗' if v.h1_degenerate_dual else '·'} "
                f"| {'✗' if v.h2_straggler else '·'} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append(_interpret(h1_rate, h2_rate, h1_rate_low, h1_rate_high))

    return "\n".join(lines) + "\n"


def _reference_solver_summary(by_inst: dict[str, list[dict]]) -> list[str]:
    """One bullet per instance describing the reference solver outcome."""
    out: list[str] = []
    any_data = False
    any_neg_gap_at_optimal = False
    for inst_id, recs in sorted(by_inst.items()):
        first = recs[0]["record"]
        status = first.get("opt_status")
        opt_meta = first.get("opt_metadata") or {}
        opt_wall = first.get("opt_wall_s")
        if status is None and not opt_meta:
            continue
        any_data = True
        highs_status = opt_meta.get("highs_status", "n/a")
        mip_gap = opt_meta.get("mip_gap")
        gap_str = f"{mip_gap:.2e}" if isinstance(mip_gap, (int, float)) else "n/a"
        deltas = [r["record"]["opt_profit"] - r["record"]["hld_profit"] for r in recs]
        worst_for_ref = -min(deltas) if min(deltas) < 0 else 0
        warn = ""
        if status == "optimal" and worst_for_ref > 0:
            any_neg_gap_at_optimal = True
            warn = (
                f" ← HLD beats reference by up to {worst_for_ref} units "
                "(HiGHS within default `mip_rel_gap`)"
            )
        elif worst_for_ref > 0:
            warn = f" ← HLD beats reference by up to {worst_for_ref} units"
        out.append(
            f"- `{inst_id}`: status=**{status}** (HiGHS={highs_status}), "
            f"mip_gap={gap_str}, ref wall={opt_wall:.1f} s{warn}\n"
        )
    if not any_data:
        return []
    if any_neg_gap_at_optimal:
        out.append(
            "\n*HiGHS returns `kOptimal` whenever the residual MIP gap is "
            "below its default `mip_rel_gap` tolerance (~1e-4). On these "
            "instances HLD's combinatorial Phase-3 occasionally yields a "
            "strictly better integer-feasible solution than HiGHS's accepted "
            "incumbent, producing tiny negative optimality gaps in the "
            "tables below. To get a hard true-optimum reference, lower "
            "`HiGHS.mip_rel_gap` (or `mip_abs_gap`) to ≤ 1e-9 in the "
            "adapter; the current values are within HiGHS's own tolerance.*\n"
        )
    else:
        out.append(
            "\n*Negative optimality gaps mean the reference solver returned "
            "a feasible-but-suboptimal incumbent (typically because "
            "`time_limit` was hit). Treat the gap as a lower bound on HLD's "
            "true gap to the exact optimum in those rows.*\n"
        )
    return out


H1_LOW_NITER_BOUNDARY: int = 15  # N_iter <= boundary is "still-narrowing bisection"


def _h1_rate_by_regime(analyses: list[dict]) -> tuple[float, float]:
    low = [a for a in analyses if a["record"]["n_iter"] <= H1_LOW_NITER_BOUNDARY]
    high = [a for a in analyses if a["record"]["n_iter"] > H1_LOW_NITER_BOUNDARY]
    low_rate = sum(1 for a in low if a["verdicts"].h1_degenerate_dual) / len(low) if low else 0.0
    high_rate = (
        sum(1 for a in high if a["verdicts"].h1_degenerate_dual) / len(high) if high else 0.0
    )
    return low_rate, high_rate


def _interpret(h1_rate: float, h2_rate: float, h1_rate_low: float, h1_rate_high: float) -> str:
    bits: list[str] = []
    if h1_rate_high >= 0.5:
        bits.append(
            "- **H1 supported.** Even at large `N_iter` "
            f"(>{H1_LOW_NITER_BOUNDARY}) Phase-1 fails to converge: the "
            "Lagrange multiplier oscillates and the selection cost flips "
            "around the budget. Phase-2 estimated costs `C_k` are therefore "
            "noisy, which would explain the non-monotonic gap-vs-N_iter "
            "behaviour in Figure 4.\n"
        )
    elif h1_rate_low >= 0.5 and h1_rate_high < 0.3:
        bits.append(
            "- **H1 not supported.** Phase-1 converges cleanly when given "
            f"enough iterations (H1 rate {h1_rate_low:.0%} for "
            f"`N_iter <= {H1_LOW_NITER_BOUNDARY}`, "
            f"{h1_rate_high:.0%} for `N_iter > {H1_LOW_NITER_BOUNDARY}`). "
            "The early-iteration H1 flags reflect the bisection's geometric "
            "narrowing window, not degeneracy of the dual basis.\n"
        )
    else:
        bits.append(
            "- H1 not supported by this run. Phase-1 converges in the majority of records.\n"
        )
    if h2_rate >= 0.5:
        bits.append(
            "- **H2 supported.** A single Phase-3 batch dominates the wall "
            "time, so equal-vs-proportional Phase-2 splits cannot move the "
            "needle on total runtime. This matches the wall-time plateau in "
            "Figure 5.\n"
        )
    else:
        bits.append(
            "- H2 not supported by this run. Phase-3 wall time is balanced across batches.\n"
        )
    return "".join(bits)


def _render_plots(analyses: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_inst: dict[str, list[SweepRecord]] = defaultdict(list)
    for a in analyses:
        by_inst[a["record"]["inst_id"]].append(a)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for inst_id, recs in sorted(by_inst.items()):
        recs.sort(key=lambda r: r["record"]["n_iter"])
        n_iters = [r["record"]["n_iter"] for r in recs]
        gaps = [r["record"]["optimality_gap"] for r in recs]
        walls = [r["record"]["hld_wall_s"] for r in recs]
        max_share = [r["verdicts"].phase3.max_batch_share for r in recs]
        spreads = [r["verdicts"].phase1.lambda_rel_spread for r in recs]

        fig, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.plot(n_iters, gaps, marker="o", label="optimality gap", color="tab:blue")
        ax1.set_xlabel("N_iter")
        ax1.set_ylabel("optimality gap", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(n_iters, walls, marker="s", label="HLD wall (s)", color="tab:red")
        ax2.set_ylabel("HLD wall time (s)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        fig.suptitle(inst_id)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{inst_id}_gap_wall.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(n_iters, spreads, marker="o", label="λ rel spread (H1)", color="tab:purple")
        ax.plot(n_iters, max_share, marker="s", label="max batch share (H2)", color="tab:green")
        ax.axhline(H1_LAMBDA_REL_SPREAD_THRESHOLD, color="tab:purple", linestyle=":", alpha=0.5)
        ax.axhline(H2_MAX_BATCH_SHARE_THRESHOLD, color="tab:green", linestyle=":", alpha=0.5)
        ax.set_xlabel("N_iter")
        ax.set_ylabel("hypothesis indicator")
        ax.legend()
        ax.set_title(f"{inst_id} — H1/H2 indicators")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{inst_id}_h1h2.pdf")
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
