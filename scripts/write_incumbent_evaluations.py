#!/usr/bin/env python3
"""Write per-instance evaluations for a completed SMAC incumbent."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from tuning.smac_run import (
    HldConfig,
    HldEvaluation,
    evaluate_hld,
    load_tuning_archive,
)

LOGGER = logging.getLogger("write_incumbent_evaluations")

FIELDNAMES = [
    "instance_id",
    "n_iter",
    "alpha",
    "k",
    "lambda_max",
    "seed",
    "profit",
    "ref_profit",
    "optimality_gap",
    "wall_time_s",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed SMAC run directory.")
    parser.add_argument(
        "--archive", type=Path, default=Path("instances"), help="Instance archive root."
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest path override.")
    parser.add_argument(
        "--incumbent-json",
        type=Path,
        default=None,
        help="Incumbent JSON path (default: <run-dir>/incumbent.json).",
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=None,
        help="Reference cache path (default: <run-dir>/reference_profits.json).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <run-dir>/incumbent_evaluations.csv).",
    )
    parser.add_argument(
        "--max-N",
        dest="max_n",
        type=int,
        default=10000,
        help="Exclude tuning instances with N above this value (default: 10000).",
    )
    parser.add_argument(
        "--max-instances", type=int, default=None, help="Cap tuning instances for testing."
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel HLD workers (default: 1).")
    parser.add_argument(
        "--seed", type=int, default=None, help="Evaluation seed (default: SMAC seed)."
    )
    parser.add_argument(
        "--eval-time-limit-s",
        type=float,
        default=60,
        help="Per-instance HLD cap in seconds (default: 60).",
    )
    parser.add_argument(
        "--ref-time-limit-s",
        type=float,
        default=60,
        help="HiGHS reference cap if the reference cache is incomplete (default: 60).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings and exit.")
    return parser.parse_args(argv)


def load_incumbent_config(path: Path) -> tuple[HldConfig, int]:
    payload = json.loads(path.read_text())
    config = payload["config"]
    seed = int(payload.get("smac", {}).get("seed", 0))
    return (
        HldConfig(
            n_iter=int(config["N_iter"]),
            alpha=float(config["alpha"]),
            k=int(config["K"]),
            lambda_max=float(config["lambda_max"]),
        ),
        seed,
    )


def completed_instance_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as fh:
        return {row["instance_id"] for row in csv.DictReader(fh) if row.get("instance_id")}


def append_evaluation(path: Path, ev: HldEvaluation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if should_write_header:
            writer.writeheader()
        writer.writerow(
            {
                "instance_id": ev.instance_id,
                "n_iter": ev.config.n_iter,
                "alpha": ev.config.alpha,
                "k": ev.config.k,
                "lambda_max": ev.config.lambda_max,
                "seed": ev.seed,
                "profit": ev.profit,
                "ref_profit": ev.ref_profit,
                "optimality_gap": ev.optimality_gap,
                "wall_time_s": ev.wall_time_s,
            }
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = parse_args(argv)
    incumbent_json = args.incumbent_json or (args.run_dir / "incumbent.json")
    reference_cache = args.reference_cache or (args.run_dir / "reference_profits.json")
    out_csv = args.out_csv or (args.run_dir / "incumbent_evaluations.csv")
    config, smac_seed = load_incumbent_config(incumbent_json)
    seed = args.seed if args.seed is not None else smac_seed

    print(f"run_dir: {args.run_dir}")
    print(f"archive: {args.archive}")
    print(f"incumbent_json: {incumbent_json}")
    print(f"reference_cache: {reference_cache}")
    print(f"out_csv: {out_csv}")
    print(
        "config: "
        f"n_iter={config.n_iter} alpha={config.alpha:.3f} "
        f"k={config.k} lambda_max={config.lambda_max:.3f}"
    )
    print(f"seed: {seed}")
    print(f"jobs: {args.jobs}")
    print(f"ref_time_limit_s: {args.ref_time_limit_s}")
    print(f"eval_time_limit_s: {args.eval_time_limit_s}")
    if args.dry_run:
        return 0

    archive = load_tuning_archive(
        archive_root=args.archive,
        manifest_path=args.manifest,
        max_instances=args.max_instances,
        max_n=args.max_n,
        reference_cache=reference_cache,
        time_limit_s=args.ref_time_limit_s,
        jobs=args.jobs,
    )
    completed_ids = completed_instance_ids(out_csv)
    pending_items = [item for item in archive.items if item.rel_path not in completed_ids]
    LOGGER.info(
        "Incumbent post-processing: %d total, %d completed, %d pending",
        len(archive.items),
        len(completed_ids),
        len(pending_items),
    )

    t0 = time.perf_counter()
    result_stream = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(
        delayed(evaluate_hld)(item, config, seed=seed, time_limit_s=args.eval_time_limit_s)
        for item in pending_items
    )
    for completed, ev in enumerate(result_stream, start=1):
        append_evaluation(out_csv, ev)
        LOGGER.info(
            "incumbent progress %d/%d elapsed=%.1fs last=%s gap=%.6f time=%.2fs",
            completed,
            len(pending_items),
            time.perf_counter() - t0,
            ev.instance_id,
            ev.optimality_gap,
            ev.wall_time_s,
        )

    LOGGER.info("Wrote incumbent evaluations to %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
