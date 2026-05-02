"""D2 calibration: HLD vs HiGHS on the hardest cells (manuscript §3.7).

The target calibration configuration is

    N=100 000, M=10, correlation=inversely_strongly,
    f in {0.1, 0.5, 0.9}, time_limit=60 s

over five seeds (the default fixed test split). The script consumes
canonical instance files from `instances/inversely_strongly/N{N}_M{M}/`
and writes one gzipped CSV row per `(instance, solver, seed)` triple
to `results/d2_calibration/<date>.csv.gz`, plus a sidecar
`<date>.host.json` so the run is fully reproducible.

A `--preview` flag swaps the target N for a smaller value (default
N=200, the largest small cell shipped with the repository) so the
harness can be exercised end-to-end before the full archive is
generated. The preview CSV is written to a separate
`results/d2_calibration/preview/` subdirectory so it never displaces
the canonical 100k-class results.

Both runs use the same solvers, the same seed list, and the same
column order, so downstream `make_tables.py` can consume them
interchangeably.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from instances.io import instance_id, load_instance
from instances.schema import CorrelationKind
from solvers import get_solver
from utils.results_schema import (
    RunResult,
    write_host_metadata,
    write_results_csv_gz,
)

EXPERIMENT_NAME = "d2_calibration"
DEFAULT_M = 10
DEFAULT_F_VALUES = (0.1, 0.5, 0.9)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_TIME_LIMIT_S = 60.0
SOLVERS = ("highs", "hld")
REFERENCE_SOLVER = "highs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archive",
        type=Path,
        default=Path("instances"),
        help="Root of the instance archive (default: instances/)",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / EXPERIMENT_NAME,
        help="Where to write the gzipped CSV (default: results/d2_calibration/)",
    )
    p.add_argument(
        "--N",
        type=int,
        default=100_000,
        help="Number of classes (default: 100000 per §3.7.1)",
    )
    p.add_argument(
        "--M",
        type=int,
        default=DEFAULT_M,
        help=f"Items per class (default: {DEFAULT_M})",
    )
    p.add_argument(
        "--f-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_F_VALUES),
        help=f"Budget tightness factors (default: {DEFAULT_F_VALUES})",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"Instance seeds (default: {DEFAULT_SEEDS})",
    )
    p.add_argument(
        "--time-limit-s",
        type=float,
        default=DEFAULT_TIME_LIMIT_S,
        help=f"Per-(instance,solver) wall-clock budget (default: {DEFAULT_TIME_LIMIT_S}s)",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Run on a smaller N (default 200) and write to a 'preview/' subdir.",
    )
    p.add_argument(
        "--preview-N",
        type=int,
        default=200,
        help="N to use when --preview is set (default: 200)",
    )
    p.add_argument(
        "--solver-seed",
        type=int,
        default=0,
        help="Solver random seed (default: 0)",
    )
    return p.parse_args()


def _resolve_target(args: argparse.Namespace) -> tuple[int, Path]:
    if args.preview:
        return args.preview_N, args.results_dir / "preview"
    return args.N, args.results_dir


def _instance_path(archive: Path, *, N: int, M: int, f: float, seed: int) -> Path:
    stem = instance_id(N=N, M=M, correlation=CorrelationKind.INVERSELY_STRONGLY, f=f, seed=seed)
    return archive / "inversely_strongly" / f"N{N}_M{M}" / f"{stem}.json.gz"


def _gap_pct(reference: int, observed: int) -> float | None:
    if reference <= 0:
        return None
    return 100.0 * (reference - observed) / reference


def main() -> int:
    args = parse_args()
    target_N, results_dir = _resolve_target(args)

    instances_to_run: list[Path] = []
    missing: list[Path] = []
    for f in args.f_values:
        for seed in args.seeds:
            path = _instance_path(args.archive, N=target_N, M=args.M, f=f, seed=seed)
            if path.exists():
                instances_to_run.append(path)
            else:
                missing.append(path)

    if missing and not instances_to_run:
        print(
            f"FAIL: no instances found at N={target_N}, M={args.M}; first missing: {missing[0]}",
            file=sys.stderr,
        )
        if not args.preview:
            print(
                "Hint: §3.7.1 requires the full benchmark archive (§2.2.1). "
                "Use --preview --preview-N 200 to validate the harness.",
                file=sys.stderr,
            )
        return 2
    if missing:
        print(f"WARN: {len(missing)} expected instance(s) missing; continuing.", file=sys.stderr)

    rows: list[RunResult] = []
    for inst_path in instances_to_run:
        inst = load_instance(inst_path)
        per_instance_results: dict[str, RunResult] = {}
        for solver_name in SOLVERS:
            solver = get_solver(solver_name)
            print(
                f"-> {inst_path.name}  solver={solver_name}  time_limit={args.time_limit_s}s",
                flush=True,
            )
            res = solver.solve(
                inst,
                time_limit_s=args.time_limit_s,
                random_seed=args.solver_seed,
            )
            per_instance_results[solver_name] = RunResult(
                experiment=EXPERIMENT_NAME,
                instance_id=inst_path.stem.removesuffix(".json"),
                N=inst.N,
                M=inst.M,
                correlation=str(inst.correlation),
                f=inst.f,
                instance_seed=inst.seed,
                solver=solver_name,
                solver_seed=args.solver_seed,
                wall_time_s=res.wall_time_s,
                profit=res.profit,
                total_cost=res.total_cost,
                n_classes_selected=res.n_classes_selected,
                status=str(res.status),
                time_limit_s=args.time_limit_s,
            )
        ref = per_instance_results.get(REFERENCE_SOLVER)
        for _solver_name, row in per_instance_results.items():
            ref_profit = ref.profit if ref is not None else None
            gap = _gap_pct(ref_profit, row.profit) if ref_profit is not None else None
            rows.append(
                RunResult(
                    **{
                        **row.__dict__,
                        "reference_solver": REFERENCE_SOLVER if ref is not None else None,
                        "reference_profit": ref_profit,
                        "optimality_gap_pct": gap,
                    }
                )
            )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    csv_path = results_dir / f"{stamp}.csv.gz"
    host_path = results_dir / f"{stamp}.host.json"
    write_results_csv_gz(csv_path, rows)
    write_host_metadata(host_path)
    print(f"OK: wrote {len(rows)} rows -> {csv_path}")
    print(f"OK: host metadata -> {host_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
