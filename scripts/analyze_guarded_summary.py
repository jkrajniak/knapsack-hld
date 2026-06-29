"""Headline guarded-vs-unguarded HLD summary from existing batch-granularity CSVs.

Unguarded HLD  : always take HLD's solution.
Guarded HLD    : take HLD only when it beats PO; else take PO  (never-lose backstop).

Reports per (correlation, f) bucket and overall:
  - unguarded: n_hurt, max_harm_pp, sum_harm_pp, sum_win_pp
  - guarded  : n_hurt (=0 by construction), max_harm_pp (=0), retained_win_pp, lost_win_pp
  - fraction of total win retained by the guard

No new solves: pairs existing PO and HLD rows per cell.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _load_cells(csv_paths: list[Path]) -> dict[tuple, dict]:
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
            cells.setdefault(key, {})[r["method"]] = float(gap)
    return {k: v for k, v in cells.items() if "po" in v and "hld" in v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    cells = _load_cells(args.csv)
    if not cells:
        print("no usable cells"); return 1

    # profit proxy = (1 - gap/100) * oracle_profit; but gaps are vs the same oracle
    # per cell, so win/harm in *gap pp* is the right currency. We sum gap-pp deltas.
    rows = []
    for key, v in cells.items():
        corr, f, seed, n, m, bs = key
        po, hld = v["po"], v["hld"]
        rows.append({"corr": corr, "f": f, "n": n, "m": m, "bs": bs,
                     "po": po, "hld": hld, "delta": hld - po,
                     "guarded_gap": min(po, hld)})

    print("=== unguarded vs guarded HLD, by (correlation, f) bucket ===")
    print(f"  {'bucket':31s} | n  | ungd: hurt maxharm  sumharm  sumwin | "
          f"gd: hurt  maxharm  retained  lost  | retain%")
    tot_ungd_win = 0.0
    tot_gd_retained = 0.0
    for k in sorted({(r["corr"], r["f"]) for r in rows}):
        rs = [r for r in rows if (r["corr"], r["f"]) == k]
        ungd_hurt = [r for r in rs if r["delta"] > 0.05]
        ungd_win = [r for r in rs if r["delta"] < -0.05]
        ungd_maxharm = max((r["delta"] for r in ungd_hurt), default=0.0)
        ungd_sumharm = sum(r["delta"] for r in ungd_hurt)
        ungd_sumwin = sum(-r["delta"] for r in ungd_win)
        # guarded: gap = min(po, hld); harm vs po = max(0, guarded_gap - po) = 0
        gd_harm = [r for r in rs if r["guarded_gap"] > r["po"] + 0.05]  # always empty
        gd_maxharm = max((r["guarded_gap"] - r["po"] for r in gd_harm), default=0.0)
        # retained win = how much HLD beat PO on cells where it did
        gd_retained = sum(r["po"] - r["guarded_gap"] for r in ungd_win)
        # lost win = HLD would have won but guard skipped? No: guard keeps HLD when
        # it wins, so lost_win = 0 on help cells. Guard only swaps HLD->PO on harm
        # cells. So retained == ungd_sumwin (guard keeps all wins, drops all harms).
        lost = ungd_sumwin - gd_retained
        tot_ungd_win += ungd_sumwin
        tot_gd_retained += gd_retained
        label = f"{k[0]} / f={k[1]:.2f}"
        print(f"  {label:31s} | {len(rs):2d} | {len(ungd_hurt):4d} {ungd_maxharm:7.2f}  "
              f"{ungd_sumharm:7.2f}  {ungd_sumwin:7.2f} | "
              f"{len(gd_harm):4d}  {gd_maxharm:7.2f}  {gd_retained:7.2f}  {lost:5.2f}  | "
              f"{(gd_retained/ungd_sumwin*100 if ungd_sumwin else 0):5.1f}")

    print(f"\n  TOTAL unguarded win = {tot_ungd_win:.2f}pp ; "
          f"guarded retained = {tot_gd_retained:.2f}pp "
          f"({tot_gd_retained/tot_ungd_win*100:.1f}%)")

    # global harm comparison
    all_hurt = [r for r in rows if r["delta"] > 0.05]
    print(f"  Unguarded: {len(all_hurt)} hurt cells, "
          f"max harm {max(r['delta'] for r in all_hurt):.2f}pp, "
          f"sum harm {sum(r['delta'] for r in all_hurt):.2f}pp")
    print(f"  Guarded  : 0 hurt cells, max harm 0.00pp, sum harm 0.00pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
