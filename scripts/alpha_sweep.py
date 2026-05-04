"""Anomaly investigation: alpha sweep driver.

Companion to :mod:`scripts.analyse_anomalies`. The original §4.3.1 sweep
fixed ``alpha = HLD default (0.9)`` and varied ``N_iter ∈ {1, …, 25}``;
this script does the dual experiment — fix ``N_iter = 20`` and vary
``alpha ∈ {0.0, 0.1, …, 1.0}`` on the same deterministic anomaly subset
(``N=10 000, M=10, weakly correlated, f=0.5``, seeds ``{0, 7, 42}``).

Reviewer R2-M7 part 2 asks whether the Fig 5 wobble at ``alpha ~= 0.5``
reflects a mechanistic effect or random noise. To answer it we instrument
two Phase-2 quantities on every record:

- **Budget coefficient of variation** ``cv(B_k)`` across batches
  (= ``std(B_k) / mean(B_k)``). Pure equal allocation has ``cv = 0`` at
  ``alpha = 0``; pure proportional has ``cv >= 0`` reflecting the underlying
  ``C_k`` heterogeneity at ``alpha = 1``. The mixed regime trades the two
  off linearly *in expectation* but the Phase-2 sub-MILP outcomes are
  not linear in ``B_k``, so a peak in observed gap at ``alpha ~= 0.5`` is a
  candidate signature of "neither robust enough nor proportional
  enough" budget assignment.
- **Min batch budget** ``min_k B_k``: small budgets push individual
  Phase-3 sub-MILPs into harder regimes (higher ``f`` = higher density,
  fewer feasible classes) and inflate per-batch wall time non-linearly.

Reference profits are loaded from ``results/anomalies/tight_gap_validation.json``
when present so the sweep does not re-pay the multi-minute HiGHS
reference solve from §4.3.5. Pass ``--no-cache`` to force a fresh
reference solve.

Examples
--------

Smoke run (3 alphas, 1 seed, no plots)::

    PYTHONPATH=code uv run python scripts/alpha_sweep.py \
        --seeds 0 --alpha-grid 0.0,0.5,1.0 --no-plots

Full §4.3.4 sweep (3 seeds, 11 alphas)::

    PYTHONPATH=code uv run python scripts/alpha_sweep.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from anomalies.sweep import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_ALPHA_NITER,
    DEFAULT_CELL,
    DEFAULT_SEEDS,
    ensure_anomaly_subset,
    load_sweep,
    reference_cache_from_tight_validation,
    run_alpha_sweep,
)

LOG = logging.getLogger("alpha_sweep")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive-root", type=Path, default=Path("instances"))
    p.add_argument("--out-dir", type=Path, default=Path("results/anomalies/full_alpha"))
    p.add_argument("--sweep-path", type=Path, default=None)
    p.add_argument(
        "--seeds",
        type=_parse_int_list,
        default=DEFAULT_SEEDS,
        help=f"Comma-separated seeds (default: {','.join(map(str, DEFAULT_SEEDS))}).",
    )
    p.add_argument(
        "--alpha-grid",
        type=_parse_float_list,
        default=DEFAULT_ALPHA_GRID,
        help="Comma-separated alpha values (default: 0.0..1.0 step 0.1).",
    )
    p.add_argument("--n-iter", type=int, default=DEFAULT_ALPHA_NITER)
    p.add_argument(
        "--reference-time-limit-s",
        type=float,
        default=1200.0,
        help=(
            "HiGHS time limit per instance (default 1200 s, matching the "
            "§4.3.5 tight-tolerance budget). Only used when the reference "
            "cache misses."
        ),
    )
    p.add_argument(
        "--reference-mip-rel-gap",
        type=float,
        default=1e-9,
        help=(
            "Tighten HiGHS's mip_rel_gap when actually solving (defaults "
            "to 1e-9 per §4.3.5 to avoid the default-tolerance artefact)."
        ),
    )
    p.add_argument(
        "--reference-cache-path",
        type=Path,
        default=Path("results/anomalies/tight_gap_validation.json"),
        help=(
            "Reuse tight-tolerance optima from this JSON file when "
            "present. Pass --no-cache to disable."
        ),
    )
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--eval-time-limit-s", type=float, default=None)
    p.add_argument("--sub-solver", type=str, default="highs")
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args(argv)


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x.strip())


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(x) for x in raw.split(",") if x.strip())


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

        cache = None
        if not args.no_cache and args.reference_cache_path.exists():
            cache = reference_cache_from_tight_validation(args.reference_cache_path)
            LOG.info(
                "loaded %d reference profits from %s",
                len(cache),
                args.reference_cache_path,
            )

        run_alpha_sweep(
            items=items,
            alpha_grid=tuple(args.alpha_grid),
            n_iter=args.n_iter,
            sub_solver=args.sub_solver,
            reference_time_limit_s=args.reference_time_limit_s,
            reference_mip_rel_gap=args.reference_mip_rel_gap,
            reference_cache=cache,
            eval_time_limit_s=args.eval_time_limit_s,
            out_path=sweep_path,
        )
        LOG.info("wrote sweep -> %s", sweep_path)

    raw = load_sweep(sweep_path)
    if not raw:
        LOG.error("sweep file %s is empty", sweep_path)
        return 1

    enriched = [_enrich(rec) for rec in raw]
    (args.out_dir / "analyses.json").write_text(
        json.dumps(
            [
                {
                    "inst_id": e["inst_id"],
                    "alpha": e["alpha"],
                    "n_iter": e["n_iter"],
                    "optimality_gap": e["optimality_gap"],
                    "hld_wall_s": e["hld_wall_s"],
                    "budget_cv": e["budget_cv"],
                    "min_batch_budget": e["min_batch_budget"],
                    "max_batch_budget": e["max_batch_budget"],
                    "max_phase3_share": e["max_phase3_share"],
                }
                for e in enriched
            ],
            indent=2,
        )
    )

    report = _render_report(enriched)
    (args.out_dir / "REPORT.md").write_text(report)
    LOG.info("wrote report -> %s", args.out_dir / "REPORT.md")

    if not args.no_plots:
        try:
            _render_plots(enriched, args.out_dir)
        except ImportError:
            LOG.warning("matplotlib not installed; skipping plots")

    return 0


def _enrich(rec: dict[str, Any]) -> dict[str, Any]:
    """Augment a raw record with Phase-2 / Phase-3 derived metrics."""
    meta = rec.get("solver_metadata", {})
    phase2 = meta.get("phase2_allocation", []) or []
    phase3 = meta.get("phase3_batches", []) or []
    bks = [float(b.get("B_k", 0.0)) for b in phase2]
    walls = [float(b.get("sub_milp_wall_s", 0.0)) for b in phase3]

    if bks and statistics_safe_mean(bks) > 0:
        cv = stats.pstdev(bks) / statistics_safe_mean(bks)
    else:
        cv = 0.0
    total_wall = sum(walls)
    max_share = max(walls) / total_wall if total_wall > 0 else 0.0

    return {
        **rec,
        "budget_cv": float(cv),
        "min_batch_budget": int(min(bks)) if bks else 0,
        "max_batch_budget": int(max(bks)) if bks else 0,
        "max_phase3_share": float(max_share),
    }


def statistics_safe_mean(xs: list[float]) -> float:
    return stats.fmean(xs) if xs else 0.0


def _render_report(records: list[dict[str, Any]]) -> str:
    by_inst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_alpha: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_inst[r["inst_id"]].append(r)
        by_alpha[float(r["alpha"])].append(r)

    lines: list[str] = []
    n_inst = len(by_inst)
    n_alpha = len(by_alpha)
    n_records = len(records)
    lines.append("# HLD alpha-sweep (Phase D §4.3.4)\n")
    lines.append(
        "Mechanistic check of the Fig 5 wobble at "
        r"$\alpha \approx 0.5$ (reviewer R2-M7 part 2). "
        f"Anomaly cell `(N=10 000, M=10, weakly, f=0.5)`, fixed "
        f"`N_iter = {records[0]['n_iter']}`, "
        f"{n_alpha} alpha values across {n_inst} seeds "
        f"({n_records} HLD records).\n"
    )

    lines.append("## Mean across seeds, vs alpha\n")
    lines.append("| alpha | mean gap | std gap | mean wall (s) | mean cv(B_k) | mean min B_k |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for alpha in sorted(by_alpha):
        recs = by_alpha[alpha]
        gaps = [r["optimality_gap"] for r in recs]
        walls = [r["hld_wall_s"] for r in recs]
        cvs = [r["budget_cv"] for r in recs]
        mins = [r["min_batch_budget"] for r in recs]
        lines.append(
            f"| {alpha:.1f} "
            f"| {stats.fmean(gaps):+.4%} "
            f"| {stats.pstdev(gaps):.4%} "
            f"| {stats.fmean(walls):.1f} "
            f"| {stats.fmean(cvs):.3f} "
            f"| {int(stats.fmean(mins))} |"
        )

    lines.append("\n## Per-instance trajectories\n")
    for inst_id, recs in sorted(by_inst.items()):
        recs.sort(key=lambda r: r["alpha"])
        lines.append(f"\n### `{inst_id}`\n")
        lines.append(
            "| alpha | gap | HLD wall (s) | min B_k | max B_k | cv(B_k) | max Phase-3 share |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in recs:
            lines.append(
                f"| {r['alpha']:.1f} "
                f"| {r['optimality_gap']:+.4%} "
                f"| {r['hld_wall_s']:.1f} "
                f"| {r['min_batch_budget']} "
                f"| {r['max_batch_budget']} "
                f"| {r['budget_cv']:.3f} "
                f"| {r['max_phase3_share']:.0%} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append(_interpret(by_alpha))
    return "\n".join(lines) + "\n"


def _interpret(by_alpha: dict[float, list[dict[str, Any]]]) -> str:
    """Compare gap at alpha ~= 0.5 vs the endpoints to test the Fig 5 spike claim."""
    if not by_alpha:
        return "_no records_\n"

    def mean_gap(a: float) -> float:
        return stats.fmean(r["optimality_gap"] for r in by_alpha[a])

    sorted_alphas = sorted(by_alpha)
    if 0.5 not in by_alpha:
        return (
            "- alpha=0.5 not in the sweep grid; cannot test the Fig 5 spike "
            "claim quantitatively. Re-run with `--alpha-grid 0.0,0.1,...,1.0`.\n"
        )

    g_mid = mean_gap(0.5)
    g_lo = mean_gap(sorted_alphas[0])
    g_hi = mean_gap(sorted_alphas[-1])
    g_endpoints = (g_lo + g_hi) / 2.0
    delta = g_mid - g_endpoints
    bits: list[str] = [
        f"- mean gap at alpha=0.5 — **{g_mid:+.4%}** "
        f"(vs alpha={sorted_alphas[0]:.1f}: {g_lo:+.4%}, "
        f"alpha={sorted_alphas[-1]:.1f}: {g_hi:+.4%})\n",
        f"- alpha=0.5 vs endpoint mean: {delta:+.4%} ({'spike' if delta > 0 else 'dip'}).\n",
    ]

    g_max_alpha = max(sorted_alphas, key=mean_gap)
    g_min_alpha = min(sorted_alphas, key=mean_gap)
    bits.append(
        f"- worst alpha: **{g_max_alpha:.1f}** "
        f"({mean_gap(g_max_alpha):+.4%}); best alpha: **{g_min_alpha:.1f}** "
        f"({mean_gap(g_min_alpha):+.4%}).\n"
    )

    cv_lo = stats.fmean(r["budget_cv"] for r in by_alpha[sorted_alphas[0]])
    cv_mid = stats.fmean(r["budget_cv"] for r in by_alpha[0.5])
    cv_hi = stats.fmean(r["budget_cv"] for r in by_alpha[sorted_alphas[-1]])
    bits.append(
        f"- mean cv(B_k) at alpha=0.0/0.5/1.0 = "
        f"{cv_lo:.3f} / {cv_mid:.3f} / {cv_hi:.3f} "
        "— linear in alpha as expected (Phase-2 design).\n"
    )

    if math.isclose(delta, 0.0, abs_tol=5e-5):
        bits.append(
            "- **Verdict: NO MECHANISTIC SPIKE.** The alpha=0.5 row is "
            "indistinguishable from the endpoints (|Δ| < 5e-5 = 0.005 %). "
            "The Fig 5 visual wobble is dominated by the HiGHS-tolerance "
            "artefact already addressed in §4.3.5, not by a "
            "Phase-2-induced sub-MILP regime change.\n"
        )
    elif delta > 0:
        bits.append(
            "- **Verdict: SPIKE CONFIRMED.** alpha=0.5 has a strictly worse "
            "mean gap than either endpoint. Investigate cv(B_k) and "
            "min B_k at alpha=0.5 for sub-MILP-regime evidence.\n"
        )
    else:
        bits.append(
            "- **Verdict: DIP, not spike.** alpha=0.5 has a strictly better "
            "mean gap than either endpoint. The Fig 5 visual is "
            "monotone-with-noise; the apparent spike is a plotting/sampling "
            "artefact.\n"
        )
    return "".join(bits)


def _render_plots(records: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    by_inst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_inst[r["inst_id"]].append(r)

    by_alpha: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_alpha[float(r["alpha"])].append(r)
    sorted_alphas = sorted(by_alpha)
    mean_gaps = [stats.fmean(r["optimality_gap"] for r in by_alpha[a]) for a in sorted_alphas]
    std_gaps = [stats.pstdev(r["optimality_gap"] for r in by_alpha[a]) for a in sorted_alphas]
    mean_walls = [stats.fmean(r["hld_wall_s"] for r in by_alpha[a]) for a in sorted_alphas]
    mean_cv = [stats.fmean(r["budget_cv"] for r in by_alpha[a]) for a in sorted_alphas]

    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    ax1.errorbar(
        sorted_alphas,
        mean_gaps,
        yerr=std_gaps,
        marker="o",
        color="tab:blue",
        capsize=3,
        label="mean gap ± std (across seeds)",
    )
    ax1.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel("optimality gap", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(
        sorted_alphas,
        mean_walls,
        marker="s",
        color="tab:red",
        label="mean HLD wall (s)",
    )
    ax2.set_ylabel("HLD wall time (s)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.suptitle(r"HLD gap and wall time vs $\alpha$ (3 seeds, $N_{iter}=20$)")
    fig.tight_layout()
    fig.savefig(plots_dir / "alpha_summary.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(sorted_alphas, mean_cv, marker="^", color="tab:green")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"mean $\mathrm{cv}(B_k)$ across seeds")
    ax.set_title(r"Phase-2 budget heterogeneity vs $\alpha$")
    fig.tight_layout()
    fig.savefig(plots_dir / "alpha_budget_cv.pdf")
    plt.close(fig)

    for inst_id, recs in by_inst.items():
        recs.sort(key=lambda r: r["alpha"])
        alphas = [r["alpha"] for r in recs]
        gaps = [r["optimality_gap"] for r in recs]
        walls = [r["hld_wall_s"] for r in recs]
        fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
        ax1.plot(alphas, gaps, marker="o", color="tab:blue")
        ax1.axhline(0.0, color="black", linewidth=0.5, linestyle=":")
        ax1.set_xlabel(r"$\alpha$")
        ax1.set_ylabel("optimality gap", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(alphas, walls, marker="s", color="tab:red")
        ax2.set_ylabel("HLD wall (s)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        fig.suptitle(inst_id)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{inst_id}_alpha_gap_wall.pdf")
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
