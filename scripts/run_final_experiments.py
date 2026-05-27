#!/usr/bin/env python3
"""Run final benchmark experiments with checked-in solver settings."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from instances.io import load_instance
from solvers import SolveResult, get_solver, validate_solution
from solvers.highs import HighsAdapter
from solvers.hld import CLASS_ORDERINGS, DEFAULT_CLASS_ORDERING, ClassOrdering, HldAdapter

LOGGER = logging.getLogger("run_final_experiments")

DEFAULT_OUT_CSV = Path("results") / "final_experiments" / "results.csv"

FIELDNAMES = [
    "instance_id",
    "subset",
    "N",
    "M",
    "correlation",
    "f",
    "seed",
    "solver",
    "status",
    "profit",
    "total_cost",
    "n_classes_selected",
    "wall_time_s",
    "n_iter",
    "alpha",
    "k",
    "lambda_max",
    "class_ordering",
    "fallback_equal_split",
    "error_message",
]


@dataclass(frozen=True)
class ExperimentEntry:
    rel_path: str
    subset: str
    seed: int
    cell: dict[str, Any]


@dataclass(frozen=True)
class HldSettings:
    n_iter: int
    alpha: float
    k: int
    lambda_max: float


@dataclass(frozen=True)
class CellSpec:
    n_items: int
    n_classes: int
    correlation: str
    f_value: float


def parse_cell_spec(value: str) -> CellSpec:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Cell must have form N,M,CORRELATION,F; got {value!r}"
        )
    n_raw, m_raw, correlation, f_raw = parts
    return CellSpec(
        n_items=int(n_raw),
        n_classes=int(m_raw),
        correlation=correlation,
        f_value=float(f_raw),
    )


def filter_entries_by_cells(
    entries: list[ExperimentEntry], cells: list[CellSpec]
) -> list[ExperimentEntry]:
    if not cells:
        return entries
    selected = set(cells)
    return [
        entry
        for entry in entries
        if CellSpec(
            n_items=int(entry.cell["N"]),
            n_classes=int(entry.cell["M"]),
            correlation=str(entry.cell["correlation"]),
            f_value=float(entry.cell["f"]),
        )
        in selected
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("instances"), help="Instance archive root."
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest path override.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hld_smac_best.json"),
        help="HLD calibration config (default: configs/hld_smac_best.json).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUT_CSV}).",
    )
    parser.add_argument(
        "--subset", default="test", choices=("test", "tuning", "all"), help="Manifest subset."
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["hld"],
        help="Solvers to run (default: hld). Use hld plus registry names such as highs.",
    )
    parser.add_argument(
        "--max-N",
        dest="max_n",
        type=int,
        default=None,
        help="Exclude instances with N above this value (default: no cap).",
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=None,
        metavar="N,M,CORRELATION,F",
        help=(
            "Restrict to one or more cell tuples (repeatable). Format: "
            "'N,M,CORRELATION,F'. If omitted, all manifest entries pass."
        ),
    )
    parser.add_argument("--max-instances", type=int, default=None, help="Cap instance count.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers (default: 1).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed passed to solvers.")
    parser.add_argument(
        "--highs-threads",
        type=int,
        default=None,
        help="Override HiGHS internal thread count for highs baseline runs.",
    )
    parser.add_argument(
        "--class-ordering",
        type=str,
        default=DEFAULT_CLASS_ORDERING,
        choices=list(CLASS_ORDERINGS),
        help=(
            "Class-ordering strategy applied before HLD partitions into K batches. "
            "Used by the §3.6 batch-ordering ablation (Task 3.3.2). "
            "When running multiple orderings, point each to a distinct --out-csv."
        ),
    )
    parser.add_argument(
        "--time-limit-s",
        type=float,
        default=60,
        help="Per-instance solver cap in seconds (default: 60).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan and exit.")
    return parser.parse_args(argv)


def load_hld_settings(path: Path) -> HldSettings:
    payload = json.loads(path.read_text())
    return HldSettings(
        n_iter=int(payload["n_iter"]),
        alpha=float(payload["alpha"]),
        k=int(payload["k"]),
        lambda_max=float(payload["lambda_max"]),
    )


def load_entries(
    *,
    archive_root: Path,
    manifest_path: Path | None,
    subset: str,
    max_n: int | None,
    max_instances: int | None,
) -> list[ExperimentEntry]:
    manifest_path = manifest_path or (archive_root / "MANIFEST.json")
    manifest = json.loads(manifest_path.read_text())
    entries: list[ExperimentEntry] = []
    for raw in manifest["files"]:
        if subset != "all" and raw.get("subset") != subset:
            continue
        if max_n is not None and int(raw["cell"]["N"]) > max_n:
            continue
        entries.append(
            ExperimentEntry(
                rel_path=str(raw["path"]),
                subset=str(raw.get("subset", "")),
                seed=int(raw["seed"]),
                cell=dict(raw["cell"]),
            )
        )
    entries.sort(key=lambda entry: entry.rel_path)
    if max_instances is not None:
        entries = entries[:max_instances]
    return entries


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as fh:
        return {
            (row["solver"], row["instance_id"])
            for row in csv.DictReader(fh)
            if row.get("solver") and row.get("instance_id")
        }


def run_one(
    *,
    archive_root: Path,
    entry: ExperimentEntry,
    solver_name: str,
    hld_settings: HldSettings,
    seed: int,
    time_limit_s: float | None,
    highs_threads: int | None = None,
    class_ordering: ClassOrdering = DEFAULT_CLASS_ORDERING,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        inst = load_instance(archive_root / entry.rel_path)
        solver = _make_solver(
            solver_name,
            hld_settings,
            highs_threads=highs_threads,
            class_ordering=class_ordering,
        )
        result = solver.solve(inst, time_limit_s=time_limit_s, random_seed=seed)
        validate_solution(inst, result)
        return _row_from_result(
            entry=entry,
            solver_name=solver_name,
            result=result,
            wall_time_s=time.perf_counter() - t0,
            hld_settings=hld_settings if solver_name == "hld" else None,
            class_ordering=class_ordering if solver_name == "hld" else None,
        )
    except Exception as exc:
        return _error_row(
            entry=entry,
            solver_name=solver_name,
            wall_time_s=time.perf_counter() - t0,
            hld_settings=hld_settings if solver_name == "hld" else None,
            error_message=f"{type(exc).__name__}: {exc}",
            class_ordering=class_ordering if solver_name == "hld" else None,
        )


def _make_solver(
    solver_name: str,
    hld_settings: HldSettings,
    *,
    highs_threads: int | None = None,
    class_ordering: ClassOrdering = DEFAULT_CLASS_ORDERING,
) -> Any:
    if solver_name == "hld":
        return HldAdapter(
            n_iter=hld_settings.n_iter,
            alpha=hld_settings.alpha,
            k=hld_settings.k,
            lambda_max_override=hld_settings.lambda_max,
            class_ordering=class_ordering,
        )
    if solver_name == "highs" and highs_threads is not None:
        return HighsAdapter(threads=highs_threads)
    return get_solver(solver_name)


def _row_from_result(
    *,
    entry: ExperimentEntry,
    solver_name: str,
    result: SolveResult,
    wall_time_s: float,
    hld_settings: HldSettings | None,
    class_ordering: ClassOrdering | None,
) -> dict[str, Any]:
    fallback = result.solver_metadata.get("fallback_equal_split")
    return {
        "instance_id": entry.rel_path,
        "subset": entry.subset,
        "N": entry.cell["N"],
        "M": entry.cell["M"],
        "correlation": entry.cell["correlation"],
        "f": entry.cell["f"],
        "seed": entry.seed,
        "solver": solver_name,
        "status": str(result.status),
        "profit": int(result.profit),
        "total_cost": int(result.total_cost),
        "n_classes_selected": int(result.n_classes_selected),
        "wall_time_s": float(wall_time_s),
        "n_iter": "" if hld_settings is None else hld_settings.n_iter,
        "alpha": "" if hld_settings is None else hld_settings.alpha,
        "k": "" if hld_settings is None else hld_settings.k,
        "lambda_max": "" if hld_settings is None else hld_settings.lambda_max,
        "class_ordering": "" if class_ordering is None else class_ordering,
        "fallback_equal_split": "" if fallback is None else int(bool(fallback)),
        "error_message": "",
    }


def _error_row(
    *,
    entry: ExperimentEntry,
    solver_name: str,
    wall_time_s: float,
    hld_settings: HldSettings | None,
    error_message: str,
    class_ordering: ClassOrdering | None,
) -> dict[str, Any]:
    return {
        "instance_id": entry.rel_path,
        "subset": entry.subset,
        "N": entry.cell["N"],
        "M": entry.cell["M"],
        "correlation": entry.cell["correlation"],
        "f": entry.cell["f"],
        "seed": entry.seed,
        "solver": solver_name,
        "status": "error",
        "profit": "",
        "total_cost": "",
        "n_classes_selected": "",
        "wall_time_s": float(wall_time_s),
        "n_iter": "" if hld_settings is None else hld_settings.n_iter,
        "alpha": "" if hld_settings is None else hld_settings.alpha,
        "k": "" if hld_settings is None else hld_settings.k,
        "lambda_max": "" if hld_settings is None else hld_settings.lambda_max,
        "class_ordering": "" if class_ordering is None else class_ordering,
        "fallback_equal_split": "",
        "error_message": error_message,
    }


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = parse_args(argv)
    hld_settings = load_hld_settings(args.config)
    cells = [parse_cell_spec(value) for value in (args.cell or [])]
    entries = filter_entries_by_cells(
        load_entries(
            archive_root=args.archive,
            manifest_path=args.manifest,
            subset=args.subset,
            max_n=args.max_n,
            max_instances=None,
        ),
        cells,
    )
    if args.max_instances is not None:
        entries = entries[: args.max_instances]

    print(f"archive: {args.archive}")
    print(f"manifest: {args.manifest or args.archive / 'MANIFEST.json'}")
    print(f"config: {args.config}")
    print(f"out_csv: {args.out_csv}")
    print(f"subset: {args.subset}")
    print(f"solvers: {' '.join(args.solvers)}")
    print(f"selected_cells: {len(cells)}")
    for cell in cells:
        print(f"cell: {cell.n_items},{cell.n_classes},{cell.correlation},{cell.f_value:g}")
    print(f"eligible_instances: {len(entries)}")
    print(f"jobs: {args.jobs}")
    print(f"highs_threads: {args.highs_threads}")
    print(f"class_ordering: {args.class_ordering}")
    print(f"time_limit_s: {args.time_limit_s}")
    if args.dry_run:
        return 0

    done = completed_keys(args.out_csv)
    work = [
        (solver_name, entry)
        for solver_name in args.solvers
        for entry in entries
        if (solver_name, entry.rel_path) not in done
    ]
    LOGGER.info(
        "Final experiments: %d planned rows (%d already complete)",
        len(work),
        len(done),
    )
    t0 = time.perf_counter()
    result_stream = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(run_one)(
            archive_root=args.archive,
            entry=entry,
            solver_name=solver_name,
            hld_settings=hld_settings,
            seed=args.seed,
            time_limit_s=args.time_limit_s,
            highs_threads=args.highs_threads,
            class_ordering=args.class_ordering,
        )
        for solver_name, entry in work
    )
    for completed, row in enumerate(result_stream, start=1):
        append_row(args.out_csv, row)
        LOGGER.info(
            "final progress %d/%d elapsed=%.1fs solver=%s instance=%s profit=%s status=%s",
            completed,
            len(work),
            time.perf_counter() - t0,
            row["solver"],
            row["instance_id"],
            row["profit"],
            row["status"],
        )

    LOGGER.info("Wrote final experiment rows to %s", args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
