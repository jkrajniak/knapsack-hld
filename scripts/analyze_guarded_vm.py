"""Summarize guarded-HLD VM sweep: quality vs PO/HLD and compute overhead.

Joins guarded_hld rows from a guarded CSV with po/hld rows from the existing
batch-granularity CSVs (same cell key). Reports:
  - never-lose: guarded gap <= po gap (should be 0 violations on clean cells)
  - win retention vs unguarded HLD
  - skip rate and median wall overhead (wall_s vs po wall from guarded metadata)
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _load(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for r in csv.DictReader(path.open(newline="")):
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
        out.setdefault(key, {})[r["method"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guarded-csv", type=Path, required=True)
    ap.add_argument("--baseline-csv", type=Path, nargs="+", required=True)
    args = ap.parse_args()

    gd_cells = _load(args.guarded_csv)
    base: dict[tuple, dict] = {}
    for p in args.baseline_csv:
        for k, v in _load(p).items():
            base.setdefault(k, {}).update(v)

    n_viol = 0
    n_cells = 0
    decisions: Counter[str] = Counter()
    gd_vs_hld_win = 0.0
    gd_vs_hld_lost = 0.0

    print("=== guarded vs PO (never-lose) ===")
    print(f"  {'bucket':31s} | n | viol | skip% | med OH | gd<=po always")
    for bucket in sorted({(k[0], k[1]) for k in gd_cells}):
        rs = [k for k in gd_cells if (k[0], k[1]) == bucket]
        viol = 0
        skips = 0
        ohs: list[float] = []
        for key in rs:
            gd = gd_cells[key].get("guarded_hld")
            po = base.get(key, {}).get("po")
            if not gd or not po:
                continue
            n_cells += 1
            gd_gap = float(gd["gap_oracle_pct"])
            po_gap = float(po["gap_oracle_pct"])
            if gd_gap > po_gap + 1e-6:
                viol += 1
                n_viol += 1
            dec = gd.get("decision", "")
            decisions[dec] += 1
            if dec == "skip":
                skips += 1
            wpo = gd.get("wall_po_s")
            whld = gd.get("wall_hld_s")
            if wpo and whld:
                ohs.append(float(wpo) / max(float(whld), 1e-9))
            hld = base.get(key, {}).get("hld")
            if hld:
                hld_gap = float(hld["gap_oracle_pct"])
                if gd_gap < hld_gap - 0.05:
                    gd_vs_hld_win += hld_gap - gd_gap
                elif gd_gap > hld_gap + 0.05:
                    gd_vs_hld_lost += gd_gap - hld_gap

        med_oh = sorted(ohs)[len(ohs) // 2] if ohs else 0.0
        skip_pct = 100.0 * skips / len(rs) if rs else 0.0
        label = f"{bucket[0]} / f={bucket[1]:.2f}"
        print(
            f"  {label:31s} | {len(rs):2d} | {viol:4d} | {skip_pct:5.1f} | "
            f"{med_oh:6.2f} | {'OK' if viol == 0 else 'FAIL'}"
        )

    print(f"\n  TOTAL cells={n_cells} violations={n_viol}")
    print(f"  decisions: {dict(decisions)}")
    print(
        f"  quality vs unguarded HLD: avoided harm {gd_vs_hld_win:.2f}pp, "
        f"extra gap vs HLD wins {gd_vs_hld_lost:.2f}pp"
    )
    return 0 if n_viol == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
