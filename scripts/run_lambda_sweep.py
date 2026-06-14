#!/usr/bin/env python3
"""λ_max diagnostic sweep on Pisinger type-1 vs type-2 (Task 1.5.2 follow-up).

The full validation grid (`run_pisinger_validation.py`) showed HLD with the
SMAC incumbent (n_iter=35, α=0.998, k=58, λ_max=80.745) attains 0.01-0.4%
median gaps on Pisinger type-1 (uncorrelated) but 5-13% medians and 0%
optimality rate on type-2 (weakly correlated).

Hypothesis: λ_max was calibrated on type-1's heavy-tailed p/w gradient
distribution. Type-2 has p ≈ w + U(-10, 10), so the per-item gradient
concentrates near 1 with small noise — the Lagrangian dual saturates at
λ scales O(1), not O(80). λ_max ≫ saturation scale wastes the
sub-gradient schedule on irrelevant λ values.

This sweep holds (n_iter, α, k) fixed at the SMAC incumbent and varies
only λ_max across a logarithmic grid, on a small set of representative
type-1 and type-2 cells. Output is a CSV one row per
(cell, seed, λ_max). Downstream: aggregate to per-(type, λ_max) median
gap and plot.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from instances.pisinger_generator import generate_pisinger_instance
from instances.schema import InstanceModel
from solvers import SolveResult, get_solver, validate_solution
from solvers.hld import HldAdapter

LOGGER = logging.getLogger("run_lambda_sweep")

DEFAULT_OUT_CSV = Path("results") / "pisinger_validation" / "lambda_sweep.csv"
DEFAULT_SMAC_CONFIG = Path("configs") / "hld_smac_best.json"

DEFAULT_LAMBDAS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.745, 160.0]
DEFAULT_CELLS = [
    # (type_id, k, n, r) — one representative from each correlation × size regime
    (1, 100, 100, 10000),
    (1, 10, 10, 10000),
    (2, 100, 100, 10000),
    (2, 10, 10, 10000),
]
DEFAULT_SEEDS = list(range(1, 21))

FIELDNAMES = [
    "instance_id",
    "type_id",
    "k",
    "n",
    "r",
    "seed",
    "lambda_max",
    "n_iter",
    "alpha",
    "hld_k",
    "hld_profit",
    "opt_profit",
    "gap_pct",
    "wall_time_s",
    "status",
    "error_message",
]


@dataclass(frozen=True)
class HldSettings:
    n_iter: int
    alpha: float
    k: int


def load_base_settings(path: Path) -> HldSettings:
    p = json.loads(path.read_text())
    return HldSettings(n_iter=int(p["n_iter"]), alpha=float(p["alpha"]), k=int(p["k"]))


def already_done(out_csv: Path) -> set[tuple[str, str]]:
    if not out_csv.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    with out_csv.open("r", newline="") as fh:
        for row in csv.DictReader(fh):
            seen.add((row["instance_id"], row["lambda_max"]))
    return seen


def solve_optimum(instance: InstanceModel, time_limit_s: float) -> int:
    """Reference profit: mcknap-optimal first, HiGHS-optimal fallback."""
    mcknap = get_solver("mcknap")
    result = mcknap.solve(instance, time_limit_s=time_limit_s)
    validate_solution(instance, result)
    if str(result.status) == "optimal":
        return int(result.profit)
    highs = get_solver("highs")
    hres = highs.solve(instance, time_limit_s=time_limit_s)
    validate_solution(instance, hres)
    if str(hres.status) != "optimal":
        raise RuntimeError(
            f"Neither mcknap ({result.status}) nor HiGHS ({hres.status}) returned optimal"
        )
    return int(hres.profit)


def solve_hld(
    instance: InstanceModel,
    base: HldSettings,
    lambda_max: float,
    time_limit_s: float,
) -> tuple[SolveResult, str]:
    solver = HldAdapter(
        n_iter=base.n_iter,
        alpha=base.alpha,
        k=base.k,
        lambda_max_override=lambda_max,
    )
    try:
        r = solver.solve(instance, time_limit_s=time_limit_s)
        try:
            validate_solution(instance, r)
        except Exception as exc:
            return r, f"validation_failed: {exc}"
        return r, ""
    except Exception as exc:
        return (
            SolveResult(profit=0, items_selected={}, total_cost=0, wall_time_s=0.0, status="error"),  # type: ignore[arg-type]
            f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--smac-config", type=Path, default=DEFAULT_SMAC_CONFIG)
    ap.add_argument(
        "--cells",
        type=str,
        nargs="+",
        default=None,
        help="Cell specs as 't<type>_k<K>_n<N>_r<R>'. Default = built-in 4 cells.",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    ap.add_argument("--time-limit-s", type=float, default=60.0)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base = load_base_settings(args.smac_config)
    LOGGER.info("Base SMAC settings (held fixed): %s", base)

    if args.cells is None:
        cells = DEFAULT_CELLS
    else:
        import re

        cells = []
        for s in args.cells:
            m = re.fullmatch(r"t(\d+)_k(\d+)_n(\d+)_r(\d+)", s)
            if not m:
                raise ValueError(f"Cell spec {s!r} must match 't<type>_k<K>_n<N>_r<R>'")
            cells.append(tuple(int(g) for g in m.groups()))  # type: ignore[arg-type]

    LOGGER.info(
        "Sweep grid: %d cells × %d seeds × %d lambdas = %d HLD runs",
        len(cells),
        len(args.seeds),
        len(args.lambdas),
        len(cells) * len(args.seeds) * len(args.lambdas),
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    seen = already_done(args.out_csv)
    if seen:
        LOGGER.info("Resume: %d (instance_id, lambda_max) rows already present", len(seen))

    write_header = not args.out_csv.exists()
    t_start = time.perf_counter()
    n_done = 0

    with args.out_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for type_id, k, n, r in cells:
            for seed in args.seeds:
                instance = generate_pisinger_instance(
                    n_classes=k,
                    n_items=n,
                    r=r,
                    type_id=type_id,
                    seed=seed,
                )
                iid = f"pisinger_t{type_id}_k{k}_n{n}_r{r}_s{seed}"
                opt_needed = any((iid, f"{lm}") not in seen for lm in args.lambdas)
                opt_profit = solve_optimum(instance, args.time_limit_s) if opt_needed else 0

                for lm in args.lambdas:
                    if (iid, str(lm)) in seen:
                        continue
                    result, err = solve_hld(instance, base, lm, args.time_limit_s)
                    gap_pct = (
                        ((opt_profit - result.profit) / opt_profit * 100)
                        if opt_profit > 0
                        else float("nan")
                    )
                    writer.writerow(
                        {
                            "instance_id": iid,
                            "type_id": type_id,
                            "k": k,
                            "n": n,
                            "r": r,
                            "seed": seed,
                            "lambda_max": lm,
                            "n_iter": base.n_iter,
                            "alpha": base.alpha,
                            "hld_k": base.k,
                            "hld_profit": result.profit,
                            "opt_profit": opt_profit,
                            "gap_pct": f"{gap_pct:.4f}",
                            "wall_time_s": f"{result.wall_time_s:.6f}",
                            "status": str(result.status),
                            "error_message": err,
                        }
                    )
                    fh.flush()
                    n_done += 1
                    if n_done % args.log_every == 0:
                        elapsed = time.perf_counter() - t_start
                        LOGGER.info("[%d runs] elapsed %.1fs", n_done, elapsed)

    LOGGER.info(
        "Done: %d new HLD runs in %.1fs -> %s", n_done, time.perf_counter() - t_start, args.out_csv
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
