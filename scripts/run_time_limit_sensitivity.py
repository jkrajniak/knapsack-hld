#!/usr/bin/env python3
"""Run HLD time-limit sensitivity checks on selected hard benchmark cells."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_final_experiments import (
    FIELDNAMES as FINAL_FIELDNAMES,
)
from run_final_experiments import (
    ExperimentEntry,
    load_entries,
    load_hld_settings,
    run_one,
)

LOGGER = logging.getLogger("run_time_limit_sensitivity")

DEFAULT_OUT_CSV = Path("results") / "final_experiments" / "time_limit_sensitivity.csv"
DEFAULT_CELLS = (
    "100000,20,inversely_strongly,0.5",
    "100000,20,strongly,0.5",
    "100000,20,uncorrelated,0.5",
    "100000,10,strongly,0.5",
    "100000,5,strongly,0.75",
    "100000,10,inversely_strongly,0.1",
)
FIELDNAMES = ["time_limit_s", *FINAL_FIELDNAMES]


@dataclass(frozen=True)
class CellSpec:
    n_items: int
    n_classes: int
    correlation: str
    f_value: float


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
        "--time-limits-s",
        nargs="+",
        type=float,
        default=[30, 60, 120, 300],
        help="Time limits to evaluate in seconds (default: 30 60 120 300).",
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=None,
        metavar="N,M,CORRELATION,F",
        help="Selected cell. Can be repeated. Defaults to six hard cells.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers (default: 1).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed passed to HLD.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan and exit.")
    return parser.parse_args(argv)


def parse_cell_specs(values: list[str] | None) -> list[CellSpec]:
    return [_parse_cell_spec(value) for value in (values or list(DEFAULT_CELLS))]


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as fh:
        return {
            (row["time_limit_s"], row["solver"], row["instance_id"])
            for row in csv.DictReader(fh)
            if row.get("time_limit_s") and row.get("solver") and row.get("instance_id")
        }


def filter_entries(entries: list[ExperimentEntry], cells: list[CellSpec]) -> list[ExperimentEntry]:
    selected = set(cells)
    return [entry for entry in entries if _entry_cell_spec(entry) in selected]


def append_sensitivity_row(path: Path, row: dict[str, Any], *, time_limit_s: float) -> None:
    row_with_limit = {"time_limit_s": _format_time_limit(time_limit_s), **row}
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row_with_limit)


def run_sensitivity_one(
    *,
    archive_root: Path,
    entry: ExperimentEntry,
    hld_settings: Any,
    seed: int,
    time_limit_s: float,
) -> tuple[float, dict[str, Any]]:
    return (
        time_limit_s,
        run_one(
            archive_root=archive_root,
            entry=entry,
            solver_name="hld",
            hld_settings=hld_settings,
            seed=seed,
            time_limit_s=time_limit_s,
        ),
    )


def _parse_cell_spec(value: str) -> CellSpec:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Cell must have form N,M,CORRELATION,F; got {value!r}")
    n_raw, m_raw, correlation, f_raw = parts
    return CellSpec(
        n_items=int(n_raw),
        n_classes=int(m_raw),
        correlation=correlation,
        f_value=float(f_raw),
    )


def _entry_cell_spec(entry: ExperimentEntry) -> CellSpec:
    return CellSpec(
        n_items=int(entry.cell["N"]),
        n_classes=int(entry.cell["M"]),
        correlation=str(entry.cell["correlation"]),
        f_value=float(entry.cell["f"]),
    )


def _format_cell(cell: CellSpec) -> str:
    return f"{cell.n_items},{cell.n_classes},{cell.correlation},{cell.f_value:g}"


def _format_time_limit(value: float) -> str:
    return f"{value:g}"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = parse_args(argv)
    cells = parse_cell_specs(args.cell)
    hld_settings = load_hld_settings(args.config)
    entries = filter_entries(
        load_entries(
            archive_root=args.archive,
            manifest_path=args.manifest,
            subset=args.subset,
            max_n=None,
            max_instances=None,
        ),
        cells,
    )
    time_limits = [_format_time_limit(value) for value in args.time_limits_s]
    planned_rows = len(entries) * len(time_limits)

    print(f"archive: {args.archive}")
    print(f"manifest: {args.manifest or args.archive / 'MANIFEST.json'}")
    print(f"config: {args.config}")
    print(f"out_csv: {args.out_csv}")
    print(f"subset: {args.subset}")
    print(f"selected_cells: {len(cells)}")
    for cell in cells:
        print(f"cell: {_format_cell(cell)}")
    print(f"eligible_instances: {len(entries)}")
    print(f"time_limits_s: {' '.join(time_limits)}")
    print(f"planned_rows: {planned_rows}")
    print(f"jobs: {args.jobs}")
    if args.dry_run:
        return 0

    done = completed_keys(args.out_csv)
    work = [
        (entry, time_limit_s)
        for time_limit_s in args.time_limits_s
        for entry in entries
        if (_format_time_limit(time_limit_s), "hld", entry.rel_path) not in done
    ]
    LOGGER.info(
        "Time-limit sensitivity: %d planned rows (%d already complete)",
        len(work),
        len(done),
    )
    t0 = time.perf_counter()
    result_stream = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(run_sensitivity_one)(
            archive_root=args.archive,
            entry=entry,
            hld_settings=hld_settings,
            seed=args.seed,
            time_limit_s=time_limit_s,
        )
        for entry, time_limit_s in work
    )
    for completed, (time_limit_s, row) in enumerate(result_stream, start=1):
        append_sensitivity_row(args.out_csv, row, time_limit_s=time_limit_s)
        LOGGER.info(
            "sensitivity progress %d/%d elapsed=%.1fs time_limit_s=%s instance=%s status=%s",
            completed,
            len(work),
            time.perf_counter() - t0,
            _format_time_limit(time_limit_s),
            row["instance_id"],
            row["status"],
        )

    LOGGER.info("Wrote time-limit sensitivity rows to %s", args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
