"""Size the guardrail's compute overhead from existing wall_s data.

Existing CSVs already log wall_s for both PO and HLD per cell (each solved with
batch_jobs=8 internal parallelism, run sequentially in the experiment). Guarded
HLD must run both, so:
  - guarded wall (serial)     ~ wall_po + wall_hld
  - overhead vs unguarded HLD = wall_po / wall_hld   (the extra cost of the safety pass)

Also reports wall_s vs batch size (bs): small bs => many small sub-MILPs.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _load(csv_paths: list[Path]) -> dict[tuple, dict]:
    cells: dict[tuple, dict] = {}
    for p in csv_paths:
        for r in csv.DictReader(p.open(newline="")):
            if int(r["n_timeout"]) > 0:
                continue
            og = r.get("oracle_gap_pct", "")
            if og not in ("", None) and abs(float(og)) > 0.5:
                continue
            key = (
                r["correlation"],
                float(r["f"]),
                int(r["seed"]),
                int(r["n"]),
                int(r["m"]),
                int(r["bs_target"]),
            )
            cells.setdefault(key, {})[r["method"]] = float(r["wall_s"])
    return {k: v for k, v in cells.items() if "po" in v and "hld" in v}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    cells = _load(args.csv)
    if not cells:
        print("no cells")
        return 1

    ratios = [v["po"] / v["hld"] for v in cells.values() if v["hld"] > 0]
    po_walls = [v["po"] for v in cells.values()]
    hld_walls = [v["hld"] for v in cells.values()]
    guarded_serial = [v["po"] + v["hld"] for v in cells.values()]

    def q(xs, p):
        xs = sorted(xs)
        return xs[int(len(xs) * p)] if xs else 0.0

    print(f"cells: {len(cells)}")
    print(
        f"wall_po   s: p25={q(po_walls, 0.25):.1f}  p50={q(po_walls, 0.5):.1f}  "
        f"p75={q(po_walls, 0.75):.1f}  p95={q(po_walls, 0.95):.1f}  max={max(po_walls):.1f}"
    )
    print(
        f"wall_hld  s: p25={q(hld_walls, 0.25):.1f}  p50={q(hld_walls, 0.5):.1f}  "
        f"p75={q(hld_walls, 0.75):.1f}  p95={q(hld_walls, 0.95):.1f}  max={max(hld_walls):.1f}"
    )
    print(
        f"guarded   s: p25={q(guarded_serial, 0.25):.1f}  p50={q(guarded_serial, 0.5):.1f}  "
        f"p75={q(guarded_serial, 0.75):.1f}  p95={q(guarded_serial, 0.95):.1f}  max={max(guarded_serial):.1f}"
    )
    print(
        f"\noverhead wall_po/wall_hld: p25={q(ratios, 0.25):.2f}  p50={q(ratios, 0.5):.2f}  "
        f"p75={q(ratios, 0.75):.2f}  p95={q(ratios, 0.95):.2f}  max={max(ratios):.2f}"
    )

    # by bs
    print("\n=== by batch size (bs): median walls and overhead ===")
    print(f"  {'bs':3s} | {'n':4s} | med_po  med_hld  med_guarded  med_overhead")
    by_bs: dict[int, list] = defaultdict(list)
    for k, v in cells.items():
        by_bs[k[5]].append(v)
    for bs in sorted(by_bs):
        vs = by_bs[bs]
        mpo = sorted(v["po"] for v in vs)[len(vs) // 2]
        mhld = sorted(v["hld"] for v in vs)[len(vs) // 2]
        mg = mpo + mhld
        oh = mpo / mhld if mhld else 0
        print(f"  {bs:3d} | {len(vs):4d} | {mpo:6.1f}  {mhld:7.1f}  {mg:11.1f}  {oh:11.2f}")

    # by N
    print("\n=== by N: median walls ===")
    print(f"  {'N':6s} | {'n':4s} | med_po  med_hld  med_guarded  med_overhead")
    by_n: dict[int, list] = defaultdict(list)
    for k, v in cells.items():
        by_n[k[3]].append(v)
    for n in sorted(by_n):
        vs = by_n[n]
        mpo = sorted(v["po"] for v in vs)[len(vs) // 2]
        mhld = sorted(v["hld"] for v in vs)[len(vs) // 2]
        mg = mpo + mhld
        oh = mpo / mhld if mhld else 0
        print(f"  {n:6d} | {len(vs):4d} | {mpo:6.1f}  {mhld:7.1f}  {mg:11.1f}  {oh:11.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
