#!/usr/bin/env python3
"""Run the Pisinger 1995 validation grid (Task 1.5.2 / R.7 LP §3.13).

Generates Pisinger-conformant instances on-the-fly via
`instances.pisinger_generator.generate_pisinger_instance(...)`, runs
the requested solvers on each, and writes one CSV row per
(instance, solver) pair. The exact `mcknap` solver acts as the
reference for optimality-gap calculations downstream.

Default grid matches Pisinger 1995 §6 verbatim:
  types: 1 (uncorrelated), 2 (weakly correlated)
  k    : {10, 100}     (number of classes)
  n    : {10, 100}     (items per class)
  r    : {1000, 10000} (coefficient range)
  seeds: 1..100        (TESTS=100 in the upstream C source)
  -> 2 * 2 * 2 * 2 * 100 = 3,200 instances per solver.

The runner is single-threaded by design: every paper experiment in this
repository uses the single-thread budget convention documented in §3.1.2
of `main.tex`. Resume is supported: rows already present in `--out-csv`
are skipped (matched on the `(instance_id, solver)` pair).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from instances.pisinger_generator import generate_pisinger_instance
from instances.schema import InstanceModel
from solvers import SolveResult, get_solver, validate_solution
from solvers.hld import HldAdapter

LOGGER = logging.getLogger("run_pisinger_validation")

DEFAULT_OUT_CSV = Path("results") / "pisinger_validation" / "results.csv"
DEFAULT_SMAC_CONFIG = Path("configs") / "hld_smac_best.json"

FIELDNAMES = [
    "instance_id",
    "type_id",
    "correlation",
    "k",
    "n",
    "r",
    "seed",
    "B",
    "solver",
    "status",
    "profit",
    "total_cost",
    "n_classes_selected",
    "wall_time_s",
    "n_iter",
    "alpha",
    "hld_k",
    "lambda_max",
    "error_message",
]


@dataclass(frozen=True)
class HldSettings:
    n_iter: int
    alpha: float
    k: int
    lambda_max: float


@dataclass(frozen=True)
class CellSpec:
    type_id: int
    k: int
    n: int
    r: int
    seed: int


def instance_id_for(cell: CellSpec) -> str:
    return f"pisinger_t{cell.type_id}_k{cell.k}_n{cell.n}_r{cell.r}_s{cell.seed}"


def load_smac_settings(path: Path) -> HldSettings:
    payload = json.loads(path.read_text())
    return HldSettings(
        n_iter=int(payload["n_iter"]),
        alpha=float(payload["alpha"]),
        k=int(payload["k"]),
        lambda_max=float(payload["lambda_max"]),
    )


def make_solver(name: str, settings: HldSettings | None):
    if name == "hld":
        if settings is None:
            raise ValueError("HLD requires SMAC settings")
        return HldAdapter(
            n_iter=settings.n_iter,
            alpha=settings.alpha,
            k=settings.k,
            lambda_max_override=settings.lambda_max,
        )
    return get_solver(name)


def build_grid(args: argparse.Namespace) -> list[CellSpec]:
    return [
        CellSpec(type_id=t, k=k, n=n, r=r, seed=s)
        for t, k, n, r, s in product(args.types, args.ks, args.ns, args.rs, args.seeds)
    ]


def already_done_rows(out_csv: Path) -> set[tuple[str, str]]:
    if not out_csv.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    with out_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            seen.add((row["instance_id"], row["solver"]))
    return seen


def run_one(
    solver_name: str,
    instance: InstanceModel,
    settings: HldSettings | None,
    time_limit_s: float,
) -> tuple[SolveResult, str]:
    """Solve `instance` with `solver_name`. Returns (result, error_message)."""
    solver = make_solver(solver_name, settings)
    try:
        result = solver.solve(instance, time_limit_s=time_limit_s)
        try:
            validate_solution(instance, result)
        except Exception as exc:
            LOGGER.warning("Solution validation failed for %s: %s", solver_name, exc)
            return result, f"validation_failed: {exc}"
        return result, ""
    except Exception as exc:
        LOGGER.exception("Solver %s raised: %s", solver_name, exc)
        return (
            SolveResult(
                profit=0,
                items_selected={},
                total_cost=0,
                wall_time_s=0.0,
                status="error",  # type: ignore[arg-type]
            ),
            f"{type(exc).__name__}: {exc}",
        )


def row_from(
    cell: CellSpec,
    instance: InstanceModel,
    solver_name: str,
    result: SolveResult,
    settings: HldSettings | None,
    error_message: str,
) -> dict[str, object]:
    n_iter = settings.n_iter if (solver_name == "hld" and settings) else ""
    alpha = settings.alpha if (solver_name == "hld" and settings) else ""
    hld_k = settings.k if (solver_name == "hld" and settings) else ""
    lambda_max = (
        result.solver_metadata.get("lambda_max", "")
        if solver_name == "hld"
        else ""
    )
    return {
        "instance_id": instance_id_for(cell),
        "type_id": cell.type_id,
        "correlation": instance.correlation.value,
        "k": cell.k,
        "n": cell.n,
        "r": cell.r,
        "seed": cell.seed,
        "B": instance.B,
        "solver": solver_name,
        "status": str(result.status),
        "profit": result.profit,
        "total_cost": result.total_cost,
        "n_classes_selected": result.n_classes_selected,
        "wall_time_s": f"{result.wall_time_s:.6f}",
        "n_iter": n_iter,
        "alpha": alpha,
        "hld_k": hld_k,
        "lambda_max": lambda_max,
        "error_message": error_message,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    ap.add_argument("--smac-config", type=Path, default=DEFAULT_SMAC_CONFIG)
    ap.add_argument("--solvers", nargs="+", default=["hld", "mcknap", "highs"])
    ap.add_argument("--types", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--ns", type=int, nargs="+", default=[10, 100])
    ap.add_argument("--rs", type=int, nargs="+", default=[1000, 10000])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 101)))
    ap.add_argument("--time-limit-s", type=float, default=60.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    settings = load_smac_settings(args.smac_config)
    LOGGER.info("HLD SMAC settings: %s", settings)

    grid = build_grid(args)
    LOGGER.info(
        "Grid: %d cells × %d solvers = %d runs",
        len(grid),
        len(args.solvers),
        len(grid) * len(args.solvers),
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    seen = already_done_rows(args.out_csv)
    if seen:
        LOGGER.info("Resume: %d (instance_id, solver) rows already present", len(seen))

    write_header = not args.out_csv.exists()
    t_start = time.perf_counter()
    runs_done = 0

    with args.out_csv.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for idx, cell in enumerate(grid, start=1):
            instance = generate_pisinger_instance(
                n_classes=cell.k,
                n_items=cell.n,
                r=cell.r,
                type_id=cell.type_id,
                seed=cell.seed,
            )
            iid = instance_id_for(cell)

            for solver_name in args.solvers:
                if (iid, solver_name) in seen:
                    continue
                solver_settings = settings if solver_name == "hld" else None
                result, error = run_one(
                    solver_name, instance, solver_settings, args.time_limit_s
                )
                writer.writerow(row_from(cell, instance, solver_name, result, solver_settings, error))
                fh.flush()
                runs_done += 1

            if idx % args.log_every == 0:
                elapsed = time.perf_counter() - t_start
                eta_s = elapsed / idx * (len(grid) - idx)
                LOGGER.info(
                    "[%d/%d] cells done | elapsed %.1fs | runs %d | ETA %.1fs",
                    idx,
                    len(grid),
                    elapsed,
                    runs_done,
                    eta_s,
                )

    LOGGER.info(
        "Completed: %d cells, %d new runs in %.1fs -> %s",
        len(grid),
        runs_done,
        time.perf_counter() - t_start,
        args.out_csv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
