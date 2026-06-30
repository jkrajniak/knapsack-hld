"""Summarize the batch-granularity sweep: is PO allocation error a function of
batch size (bs) alone, independent of N?

Reads the check_batch_granularity.py CSV and prints, per (correlation, f):
  - PO error vs bs, one column per N (curves should overlap if bs-governed);
  - the cross-N spread of PO error at each bs (small => N-independent);
  - HLD error at the same cells (recovery is largest at small bs).

Only clean rows are used for the allocation conclusion: n_timeout == 0 and a
tight oracle (oracle_gap_pct <= --max-oracle-gap). Dropped rows are reported.

Usage:
    uv run python scripts/analyze_batch_granularity.py \
        --csv results/batch_granularity/inversely_strongly.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--max-oracle-gap",
        type=float,
        default=0.5,
        help="drop rows whose oracle dual gap exceeds this (pct)",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.open(newline="")))
    # mean error over seeds, keyed by (f, method, bs, n)
    cells: dict[tuple, list[float]] = defaultdict(list)
    dropped = 0
    ns: set[int] = set()
    bss: set[int] = set()
    fs: set[float] = set()
    for r in rows:
        if int(r["n_timeout"]) > 0:
            dropped += 1
            continue
        og = r["oracle_gap_pct"]
        if og not in ("", None) and abs(float(og)) > args.max_oracle_gap:
            dropped += 1
            continue
        f = float(r["f"])
        n = int(r["n"])
        bs = int(r["bs_target"])
        cells[(f, r["method"], bs, n)].append(float(r["gap_oracle_pct"]))
        ns.add(n)
        bss.add(bs)
        fs.add(f)

    n_sorted = sorted(ns)
    bs_sorted = sorted(bss)
    print(
        f"clean rows: {len(rows) - dropped}/{len(rows)} "
        f"(dropped {dropped}: timeout or loose oracle)\n"
    )

    for f in sorted(fs):
        for method in ("po", "hld"):
            header = "  bs |" + "".join(f" N={n:>6d}" for n in n_sorted) + " | spread"
            print(f"=== f={f}  {method.upper()} error vs batch size (mean over seeds, %) ===")
            print(header)
            print("  " + "-" * (len(header) - 2))
            for bs in bs_sorted:
                vals = {
                    n: mean(cells[(f, method, bs, n)])
                    for n in n_sorted
                    if (f, method, bs, n) in cells
                }
                if not vals:
                    continue
                cols = "".join(f" {vals[n]:6.2f}" if n in vals else "      ." for n in n_sorted)
                spread = max(vals.values()) - min(vals.values())
                print(f"  {bs:3d} |{cols} | {spread:6.2f}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
