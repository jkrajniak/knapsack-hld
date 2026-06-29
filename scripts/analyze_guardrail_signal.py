"""Offline check: does PO's gap to the oracle predict whether HLD helps or hurts?

The guardrail concept: skip HLD when equal-split (PO) is already near the optimum
ceiling, because there is no allocation error left to recover. The cheapest production
signal for that is the Lagrangian upper bound vs PO's profit; here we proxy it with
the oracle gap already logged in the batch-granularity CSVs to validate the concept
*before* instrumenting HLD.

Per cell (correlation, f, seed, n, m, bs) we pair PO and HLD rows and compute
  delta = hld_gap - po_gap     (negative => HLD helped; positive => HLD hurt)

Then we ask: does po_gap separate help-cells from hurt-cells? We report, over a
range of thresholds tau:
  - among cells with po_gap < tau  (guardrail would SKIP HLD): how many did HLD
    actually hurt by more than a tolerance, and how much total harm would be
    avoided;
  - among cells with po_gap >= tau (guardrail would RUN HLD): how many did HLD
    actually help, and how much total gain would be retained.

Usage:
    uv run python scripts/analyze_guardrail_signal.py \
        --csv results/batch_granularity/inversely_strongly.csv \
        results/batch_granularity/strongly.csv \
        results/batch_granularity/mrobust_inversely_strongly_M5.csv \
        results/batch_granularity/mrobust_inversely_strongly_M20.csv \
        results/batch_granularity/mrobust_strongly_M5.csv \
        results/batch_granularity/mrobust_strongly_M20.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _load_cells(csv_paths: list[Path]) -> dict[tuple, dict]:
    """{(corr,f,seed,n,m,bs): {"po": po_gap, "hld": hld_gap, ...}} from CSVs.

    Only clean cells (both PO and HLD present, no timeout, tight oracle) are kept,
    matching the analyze_batch_granularity.py filter.
    """
    cells: dict[tuple, dict] = {}
    for p in csv_paths:
        for r in csv.DictReader(p.open(newline="")):
            if int(r["n_timeout"]) > 0:
                continue
            og = r.get("oracle_gap_pct", "")
            if og not in ("", None) and abs(float(og)) > 0.5:
                continue
            key = (r["correlation"], float(r["f"]), int(r["seed"]),
                   int(r["n"]), int(r["m"]), int(r["bs_target"]))
            gap = r.get("gap_oracle_pct")
            if gap in ("", None):
                continue
            cell = cells.setdefault(key, {})
            cell[r["method"]] = float(gap)
    return {k: v for k, v in cells.items() if "po" in v and "hld" in v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, nargs="+", required=True)
    ap.add_argument("--hurt-tol", type=float, default=0.05,
                    help="delta above this (pp) counts as a real HLD harm (default 0.05)")
    args = ap.parse_args()

    cells = _load_cells(args.csv)
    if not cells:
        print("no usable cells"); return 1

    rows = []
    for key, v in cells.items():
        corr, f, seed, n, m, bs = key
        po, hld = v["po"], v["hld"]
        delta = hld - po
        rows.append({"key": key, "corr": corr, "f": f, "n": n, "m": m, "bs": bs,
                     "po": po, "hld": hld, "delta": delta,
                     "helped": delta < -args.hurt_tol,
                     "hurt": delta > args.hurt_tol})

    rows.sort(key=lambda r: r["po"])
    n_help = sum(r["helped"] for r in rows)
    n_hurt = sum(r["hurt"] for r in rows)
    print(f"cells: {len(rows)} total | HLD helped: {n_help} | HLD hurt: {n_hurt} | "
          f"within tol: {len(rows)-n_help-n_hurt}\n")

    print("=== worst 15 HLD-harm cells (delta = hld-po, pp) ===")
    print("  corr                    f    N      M  bs |   po%   hld%  delta")
    for r in sorted(rows, key=lambda x: -x["delta"])[:15]:
        print(f"  {r['corr']:23s} {r['f']:.2f} {r['n']:6d} {r['m']:3d} {r['bs']:3d} | "
              f"{r['po']:6.2f} {r['hld']:6.2f} {r['delta']:+6.2f}")

    print("\n=== best 15 HLD-win cells ===")
    print("  corr                    f    N      M  bs |   po%   hld%  delta")
    for r in sorted(rows, key=lambda x: x["delta"])[:15]:
        print(f"  {r['corr']:23s} {r['f']:.2f} {r['n']:6d} {r['m']:3d} {r['bs']:3d} | "
              f"{r['po']:6.2f} {r['hld']:6.2f} {r['delta']:+6.2f}")

    print("\n=== guardrail sweep: skip HLD when po_gap < tau ===")
    print("  tau | skip  run | skip_but_hurt run_lost  | harm_avoided  gain_retained")
    print("       |  cells cells |   n  max_harm   n  max_lost  |     sum_pp      sum_pp")
    for tau in [0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        skip = [r for r in rows if r["po"] < tau]
        run = [r for r in rows if r["po"] >= tau]
        skip_hurt = [r for r in skip if r["hurt"]]
        run_lost = [r for r in run if r["hurt"]]
        harm_avoided = sum(r["delta"] for r in skip_hurt)  # pp of harm not incurred
        gain_retained = sum(-r["delta"] for r in run if r["helped"])  # pp of win kept
        max_skip_harm = max((r["delta"] for r in skip_hurt), default=0.0)
        max_run_lost = max((r["delta"] for r in run_lost), default=0.0)
        print(f"  {tau:4.2f} | {len(skip):5d} {len(run):5d} | "
              f"{len(skip_hurt):3d} {max_skip_harm:7.2f}  "
              f"{len(run_lost):3d} {max_run_lost:7.2f}  | "
              f"{harm_avoided:9.2f}  {gain_retained:9.2f}")

    print("\n=== with never-lose backstop (accept HLD only if profit>=PO): ===")
    print("  residual harm after backstop = 0 by construction; report retained win.")
    for tau in [0.5, 1.0, 2.0]:
        run = [r for r in rows if r["po"] >= tau]
        retained = sum(-r["delta"] for r in run if r["helped"])
        skipped_win = sum(-r["delta"] for r in rows if r["po"] < tau and r["helped"])
        print(f"  tau={tau:4.2f}: run HLD on {len(run)} cells, retain {retained:.2f}pp "
              f"of win; skip {len(rows)-len(run)} cells (would have forgone "
              f"{skipped_win:.2f}pp of win that HLD would have gained).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
