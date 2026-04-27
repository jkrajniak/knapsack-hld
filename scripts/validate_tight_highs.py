"""Before/after validation for §4.3.5 (tight `HiGHS.mip_rel_gap`).

Solves each anomaly instance twice: once with HiGHS at its default tolerance
(~1e-4) and once with `mip_rel_gap=1e-9`. Cross-references against the best
HLD profit observed in the existing full sweep at any `N_iter`. Writes a
JSON summary to `results/anomalies/tight_gap_validation.json` and prints a
markdown table.

This is a focused proof-of-concept of the change introduced in
`HighsAdapter(mip_rel_gap=...)` and `run_sweep(reference_mip_rel_gap=...)`;
it is much cheaper than re-running the full 75-record sweep but still
demonstrates the negative-gap artefact disappearing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from anomalies.sweep import DEFAULT_CELL, ensure_anomaly_subset
from solvers.highs import HighsAdapter

OUT_PATH = ROOT / "results/anomalies/tight_gap_validation.json"
SWEEP_PATH = ROOT / "results/anomalies/full/sweep.jsonl"
TIME_LIMIT_S = 1200.0  # tight gap can need >2x the default's wall time
SEEDS = (0, 7, 42)


def _best_hld_profit(sweep_path: Path) -> dict[str, int]:
    """Best HLD profit across all (inst, N_iter) records in a prior sweep."""
    best: dict[str, int] = {}
    for line in sweep_path.read_text().splitlines():
        rec = json.loads(line)
        cur = best.get(rec["inst_id"], -1)
        if rec["hld_profit"] > cur:
            best[rec["inst_id"]] = int(rec["hld_profit"])
    return best


def _solve(adapter: HighsAdapter, inst, label: str) -> dict[str, object]:
    t0 = time.perf_counter()
    res = adapter.solve(inst, time_limit_s=TIME_LIMIT_S)
    wall = time.perf_counter() - t0
    return {
        "label": label,
        "profit": int(res.profit),
        "status": str(res.status),
        "wall_s": float(wall),
        "highs_status": res.solver_metadata.get("highs_status"),
        "mip_gap": res.solver_metadata.get("mip_gap"),
        "mip_rel_gap_set": res.solver_metadata.get("mip_rel_gap_set"),
    }


def main() -> int:
    items = ensure_anomaly_subset(
        archive_root=ROOT / "instances",
        cell=DEFAULT_CELL,
        seeds=SEEDS,
    )
    best_hld = _best_hld_profit(SWEEP_PATH)

    out: list[dict[str, object]] = []
    for item in items:
        print(f"-- {item.inst_id}", flush=True)
        default_solve = _solve(HighsAdapter(), item.inst, "default")
        print(
            f"   default: profit={default_solve['profit']} wall={default_solve['wall_s']:.1f}s "
            f"mip_gap={default_solve['mip_gap']}",
            flush=True,
        )
        tight_solve = _solve(HighsAdapter(mip_rel_gap=1e-9), item.inst, "tight_1e-9")
        print(
            f"   tight  : profit={tight_solve['profit']} wall={tight_solve['wall_s']:.1f}s "
            f"mip_gap={tight_solve['mip_gap']}",
            flush=True,
        )
        hld_best = best_hld.get(item.inst_id)
        out.append(
            {
                "inst_id": item.inst_id,
                "default": default_solve,
                "tight": tight_solve,
                "hld_best_profit": hld_best,
                "hld_advantage_default": (
                    hld_best - int(default_solve["profit"]) if hld_best is not None else None
                ),
                "hld_advantage_tight": (
                    hld_best - int(tight_solve["profit"]) if hld_best is not None else None
                ),
            }
        )

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_PATH}\n")

    print("| instance | HiGHS default | HiGHS tight (1e-9) | best HLD | "
          "HLD-default | HLD-tight |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in out:
        print(
            f"| `{r['inst_id']}` | {r['default']['profit']} "
            f"({r['default']['wall_s']:.1f}s) "
            f"| {r['tight']['profit']} ({r['tight']['wall_s']:.1f}s) "
            f"| {r['hld_best_profit']} "
            f"| {r['hld_advantage_default']:+d} "
            f"| {r['hld_advantage_tight']:+d} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
