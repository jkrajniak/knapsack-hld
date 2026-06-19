#!/usr/bin/env python3
"""Batch-count (K) robustness sweep for HLD vs Partition-Optimal.

Motivation
----------
The batch count ``K`` is HLD's parallelism knob, and its effect on
solution quality is an empirical question rather than a given. This
driver makes the dependence falsifiable: it fixes the SMAC-calibrated
HLD configuration (``n_iter``, ``alpha``, ``lambda_max`` from
``configs/hld_smac_best.json``) and varies only ``K`` over a grid,
running both HLD and the equal-split Partition-Optimal reference at each K
on the same instances. For every (solver, instance, K) we record solution
profit, wall time, and HLD's Phase-2 fallback flag, so downstream analysis
can report the paired HLD-vs-equal-split gain and the HLD wall-time trend
as a function of K.

The calibrated incumbent uses ``K = 58``; the default grid
``{2, 4, 8, 16, 32, 64, 100}`` brackets it and extends to the
"up-to-100-batches" regime. Analyse the output with
``scripts/summarize_k_sweep.py``, which reports the feasible-K regime and
the per-cell sweet spot rather than assuming monotone behaviour.

Examples
--------
Smoke run (3 K values, the fast N=10000 cell)::

    PYTHONPATH=code uv run python scripts/k_sweep.py \
        --k-grid 2,8,100 --cell 10000,10,weakly,0.5 --out-jsonl results/k_sweep/smoke.jsonl

Full sweep (default grid, all eligible large-N instances)::

    PYTHONPATH=code uv run python scripts/k_sweep.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from heuristics.partition_optimal import PartitionOptimalAdapter
from instances.io import load_instance

# run_final_experiments holds the manifest-loading + cell-filtering machinery
# we deliberately reuse rather than duplicate.
from run_final_experiments import (  # type: ignore[import-not-found]
    ExperimentEntry,
    HldSettings,
    filter_entries_by_cells,
    load_entries,
    load_hld_settings,
    parse_cell_spec,
)
from solvers import SolveResult, validate_solution
from solvers.hld import HldAdapter

LOGGER = logging.getLogger("k_sweep")

DEFAULT_OUT_JSONL = Path("results") / "k_sweep" / "sweep.jsonl"
DEFAULT_K_GRID = (2, 4, 8, 16, 32, 64, 100)
DEFAULT_SOLVERS = ("hld", "partition_optimal")


def _parse_int_list(value: str) -> list[int]:
    return [int(tok) for tok in value.split(",") if tok.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("instances"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/hld_smac_best.json"))
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument(
        "--subset", default="all", choices=("test", "tuning", "all"), help="Manifest subset."
    )
    parser.add_argument(
        "--k-grid",
        type=_parse_int_list,
        default=list(DEFAULT_K_GRID),
        help=f"Comma-separated K values (default: {','.join(map(str, DEFAULT_K_GRID))}).",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=list(DEFAULT_SOLVERS),
        help="Solvers to run at each K (default: hld partition_optimal).",
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=None,
        metavar="N,M,CORRELATION,F",
        help="Restrict to one or more cell tuples (repeatable). Default: N>=10000 cells.",
    )
    parser.add_argument(
        "--min-N",
        dest="min_n",
        type=int,
        default=10000,
        help="Keep only instances with N >= this (default: 10000).",
    )
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers (default: 1).")
    parser.add_argument("--seed", type=int, default=7, help="Solver random seed (default: 7).")
    parser.add_argument("--time-limit-s", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def make_solver(solver_name: str, *, k: int, hld: HldSettings) -> Any:
    if solver_name == "hld":
        return HldAdapter(
            n_iter=hld.n_iter,
            alpha=hld.alpha,
            k=k,
            lambda_max_override=hld.lambda_max,
        )
    if solver_name == "partition_optimal":
        return PartitionOptimalAdapter(k=k)
    raise ValueError(f"unsupported solver for K-sweep: {solver_name!r}")


def record_from_result(
    *,
    entry: ExperimentEntry,
    solver_name: str,
    k: int,
    result: SolveResult,
    wall_time_s: float,
    solver_seed: int,
    time_limit_s: float,
    hld: HldSettings | None,
) -> dict[str, Any]:
    meta = result.solver_metadata or {}
    fallback = meta.get("fallback_equal_split")
    return {
        "instance_id": entry.rel_path,
        "subset": entry.subset,
        "N": int(entry.cell["N"]),
        "M": int(entry.cell["M"]),
        "correlation": str(entry.cell["correlation"]),
        "f": float(entry.cell["f"]),
        "seed": int(entry.seed),
        "solver": solver_name,
        "k": int(k),
        "status": str(result.status),
        "profit": int(result.profit),
        "total_cost": int(result.total_cost),
        "n_classes_selected": int(result.n_classes_selected),
        "wall_time_s": float(wall_time_s),
        "solver_seed": int(solver_seed),
        "time_limit_s": float(time_limit_s),
        "n_iter": None if hld is None else hld.n_iter,
        "alpha": None if hld is None else hld.alpha,
        "lambda_max": None if hld is None else hld.lambda_max,
        "fallback_equal_split": None if fallback is None else int(bool(fallback)),
        "error_message": "",
    }


def error_record(
    *,
    entry: ExperimentEntry,
    solver_name: str,
    k: int,
    wall_time_s: float,
    solver_seed: int,
    time_limit_s: float,
    error_message: str,
) -> dict[str, Any]:
    return {
        "instance_id": entry.rel_path,
        "subset": entry.subset,
        "N": int(entry.cell["N"]),
        "M": int(entry.cell["M"]),
        "correlation": str(entry.cell["correlation"]),
        "f": float(entry.cell["f"]),
        "seed": int(entry.seed),
        "solver": solver_name,
        "k": int(k),
        "status": "error",
        "profit": None,
        "total_cost": None,
        "n_classes_selected": None,
        "wall_time_s": float(wall_time_s),
        "solver_seed": int(solver_seed),
        "time_limit_s": float(time_limit_s),
        "n_iter": None,
        "alpha": None,
        "lambda_max": None,
        "fallback_equal_split": None,
        "error_message": error_message,
    }


def run_one(
    *,
    archive_root: Path,
    entry: ExperimentEntry,
    solver_name: str,
    k: int,
    hld: HldSettings,
    solver_seed: int,
    time_limit_s: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        inst = load_instance(archive_root / entry.rel_path)
        solver = make_solver(solver_name, k=k, hld=hld)
        result = solver.solve(inst, time_limit_s=time_limit_s, random_seed=solver_seed)
        validate_solution(inst, result)
        return record_from_result(
            entry=entry,
            solver_name=solver_name,
            k=k,
            result=result,
            wall_time_s=time.perf_counter() - t0,
            solver_seed=solver_seed,
            time_limit_s=time_limit_s,
            hld=hld if solver_name == "hld" else None,
        )
    except Exception as exc:
        return error_record(
            entry=entry,
            solver_name=solver_name,
            k=k,
            wall_time_s=time.perf_counter() - t0,
            solver_seed=solver_seed,
            time_limit_s=time_limit_s,
            error_message=f"{type(exc).__name__}: {exc}",
        )


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    done: set[tuple[str, str, int]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((str(row["solver"]), str(row["instance_id"]), int(row["k"])))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def write_meta(
    *,
    out_jsonl: Path,
    args: argparse.Namespace,
    hld: HldSettings,
    entries: list[ExperimentEntry],
    manifest_path: Path,
) -> Path:
    meta_path = out_jsonl.with_name(out_jsonl.stem + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "git_commit": git_commit(),
        "command": ["python", "scripts/k_sweep.py", *(sys.argv[1:])],
        "config_path": str(args.config),
        "hld_config": {
            "n_iter": hld.n_iter,
            "alpha": hld.alpha,
            "lambda_max": hld.lambda_max,
        },
        "k_grid": list(args.k_grid),
        "solvers": list(args.solvers),
        "time_limit_s": float(args.time_limit_s),
        "solver_seed": int(args.seed),
        "subset": args.subset,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_of(manifest_path),
        "instances": sorted(e.rel_path for e in entries),
        "instance_seeds": sorted({int(e.seed) for e in entries}),
        "n_instances": len(entries),
        "n_runs_planned": len(entries) * len(args.k_grid) * len(args.solvers),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = parse_args(argv)
    hld = load_hld_settings(args.config)
    cells = [parse_cell_spec(value) for value in (args.cell or [])]
    entries = filter_entries_by_cells(
        load_entries(
            archive_root=args.archive,
            manifest_path=args.manifest,
            subset=args.subset,
            max_n=None,
            max_instances=None,
        ),
        cells,
    )
    entries = [e for e in entries if int(e.cell["N"]) >= args.min_n]
    if args.max_instances is not None:
        entries = entries[: args.max_instances]

    manifest_path = args.manifest or (args.archive / "MANIFEST.json")
    n_runs = len(entries) * len(args.k_grid) * len(args.solvers)
    print(f"archive: {args.archive}")
    print(f"config: {args.config} (n_iter={hld.n_iter}, alpha={hld.alpha}, lambda_max={hld.lambda_max})")
    print(f"out_jsonl: {args.out_jsonl}")
    print(f"subset: {args.subset}  min_N: {args.min_n}")
    print(f"k_grid: {args.k_grid}")
    print(f"solvers: {' '.join(args.solvers)}")
    print(f"eligible_instances: {len(entries)}")
    print(f"time_limit_s: {args.time_limit_s}")
    print(f"runs_planned: {n_runs}")
    if args.dry_run:
        for e in entries:
            print(f"  {e.rel_path} (N={e.cell['N']}, {e.cell['correlation']}, f={e.cell['f']}, seed={e.seed})")
        return 0

    write_meta(out_jsonl=args.out_jsonl, args=args, hld=hld, entries=entries, manifest_path=manifest_path)

    done = completed_keys(args.out_jsonl)
    work = [
        (solver_name, entry, k)
        for solver_name in args.solvers
        for entry in entries
        for k in args.k_grid
        if (solver_name, entry.rel_path, int(k)) not in done
    ]
    LOGGER.info("K-sweep: %d planned runs (%d already complete)", len(work), len(done))

    t0 = time.perf_counter()
    result_stream = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(run_one)(
            archive_root=args.archive,
            entry=entry,
            solver_name=solver_name,
            k=k,
            hld=hld,
            solver_seed=args.seed,
            time_limit_s=args.time_limit_s,
        )
        for solver_name, entry, k in work
    )
    for completed, row in enumerate(result_stream, start=1):
        append_jsonl(args.out_jsonl, row)
        LOGGER.info(
            "k-sweep %d/%d elapsed=%.1fs solver=%s k=%s N=%s corr=%s f=%s seed=%s profit=%s wall=%.2fs status=%s",
            completed,
            len(work),
            time.perf_counter() - t0,
            row["solver"],
            row["k"],
            row["N"],
            row["correlation"],
            row["f"],
            row["seed"],
            row["profit"],
            row["wall_time_s"],
            row["status"],
        )

    LOGGER.info("Wrote K-sweep records to %s", args.out_jsonl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
