"""Where does HLD help vs hurt? Bucket the clean cells by (correlation, f) and by
PO fill_pct, to see whether the discriminator is clean and whether fill_pct (a
production-cheap proxy for budget looseness) separates help/hurt better than po_gap.

Reads the same batch-granularity CSVs as analyze_guardrail_signal.py.
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
            fill = r.get("fill_pct")
            if gap in ("", None):
                continue
            cell = cells.setdefault(key, {})
            cell[r["method"]] = {
                "gap": float(gap),
                "fill": float(fill) if fill not in ("", None) else None,
            }
    return {k: v for k, v in cells.items() if "po" in v and "hld" in v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, nargs="+", required=True)
    ap.add_argument("--hurt-tol", type=float, default=0.05)
    args = ap.parse_args()

    cells = _load_cells(args.csv)
    if not cells:
        print("no usable cells"); return 1

    rows = []
    for key, v in cells.items():
        corr, f, seed, n, m, bs = key
        po, hld = v["po"], v["hld"]
        delta = hld["gap"] - po["gap"]
        rows.append({"key": key, "corr": corr, "f": f, "n": n, "m": m, "bs": bs,
                     "po_gap": po["gap"], "po_fill": po["fill"],
                     "hld_gap": hld["gap"], "delta": delta,
                     "helped": delta < -args.hurt_tol, "hurt": delta > args.hurt_tol})

    print("=== bucket by (correlation, f): help / hurt / neutral, mean delta ===")
    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        buckets[(r["corr"], r["f"])].append(r)
    print(f"  {'corr':23s} {'f':4s} |  n  help hurt neut | mean_delta  min_delta  max_delta | po_fill_range")
    for k in sorted(buckets):
        rs = buckets[k]
        nh = sum(r["helped"] for r in rs)
        nu = sum(r["hurt"] for r in rs)
        nn = len(rs) - nh - nu
        md = sum(r["delta"] for r in rs) / len(rs)
        mind = min(r["delta"] for r in rs)
        maxd = max(r["delta"] for r in rs)
        fills = [r["po_fill"] for r in rs if r["po_fill"] is not None]
        fr = f"{min(fills):.1f}-{max(fills):.1f}" if fills else "-"
        print(f"  {k[0]:23s} {k[1]:.2f} | {len(rs):3d}  {nh:4d} {nu:4d} {nn:4d} | "
              f"{md:+9.2f}  {mind:+9.2f}  {maxd:+9.2f} | {fr}")

    print("\n=== fill_pct as a skip-signal: skip HLD when po_fill >= fill_thresh ===")
    print("  (high fill = loose budget = expect HLD to harm)")
    print("  fill | skip  run | skip_but_hurt run_lost  | harm_avoided  win_retained")
    for ft in [60, 70, 80, 85, 90, 92, 95, 97, 99, 100]:
        skip = [r for r in rows if r["po_fill"] is not None and r["po_fill"] >= ft]
        run = [r for r in rows if r["po_fill"] is not None and r["po_fill"] < ft]
        sh = [r for r in skip if r["hurt"]]
        rl = [r for r in run if r["hurt"]]
        ha = sum(r["delta"] for r in sh)
        wr = sum(-r["delta"] for r in run if r["helped"])
        mh = max((r["delta"] for r in sh), default=0.0)
        ml = max((r["delta"] for r in rl), default=0.0)
        print(f"  {ft:4d} | {len(skip):5d} {len(run):5d} | {len(sh):3d} {mh:7.2f}  "
              f"{len(rl):3d} {ml:7.2f}  | {ha:9.2f}  {wr:9.2f}")

    print("\n=== cross-check: among HLD-hurt cells, what is their (corr, f, fill)? ===")
    print(f"  {'corr':23s} {'f':4s} {'m':3s} {'bs':3s} | po_fill  po_gap  hld_gap  delta")
    for r in sorted(rows, key=lambda x: -x["delta"])[:25]:
        if r["hurt"]:
            print(f"  {r['corr']:23s} {r['f']:.2f} {r['m']:3d} {r['bs']:3d} | "
                  f"{r['po_fill']:6.1f}  {r['po_gap']:6.2f}  {r['hld_gap']:6.2f}  {r['delta']:+6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
