#!/usr/bin/env python3
"""Replay HLD Phase-1 + Phase-2 only on the pinned test-subset manifest.

Task 3.4.3 follow-up of 3.4.1. The pinned final-experiment archive
predates the `fallback_equal_split` CSV column, so
`summarize_fallback_stats.py` returns `n_rows = 0` when pointed at it.
This script reconstructs the fallback flag for every instance the
runner would have processed -- cheaply, by re-running only the
Phase-1 binary search and the Phase-2 estimate (no sub-MILP) -- and
emits a CSV with the same column layout
`summarize_fallback_stats.py` already consumes.

Bit-for-bit reproducibility of the pinned archive is *not* attempted:
the probe only re-derives `fallback_equal_split`, which is a
deterministic function of `(instance, n_iter, alpha, K, lambda_max,
class_ordering)`. The defaults match the paper-wide configuration
(`configs/hld_smac_best.json`, `class_ordering=sequential`).

Typical use (must run on the remote VM where the pinned manifest
lives -- the local `knapsack-hld/instances/MANIFEST.json` is a smaller
development subset of 131 test entries, whereas the pinned final-
experiments archive was generated against the VM manifest of 6300
test entries):

    uv run python scripts/replay_hld_fallback.py \
        --archive instances \
        --manifest instances/MANIFEST.json \
        --subset test \
        --config configs/hld_smac_best.json \
        --out-csv results/final_experiments/fallback_stats_pinned.csv

    uv run python scripts/summarize_fallback_stats.py \
        --results-csv results/final_experiments/fallback_stats_pinned.csv \
        --out-dir results/final_experiments/fallback_stats_pinned_summary

Expected wall-time on the M4 Pro at one job (no joblib): roughly an
hour for the full 6300-entry test subset, dominated by the ~2k
N = 100 000 instances (load + Phase-1 binary search). The script is
trivially parallelisable with joblib if that ever becomes a bottleneck;
left serial for now to keep the dependency surface small.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from instances.io import load_instance
from run_final_experiments import HldSettings, load_entries, load_hld_settings
from solvers.hld import (
    CLASS_ORDERINGS,
    DEFAULT_CLASS_ORDERING,
    ClassOrdering,
    _class_order,
    _instance_dependent_lambda_max,
    _phase1_binary_search,
    _phase2_estimate,
    _split_classes,
)

LOGGER = logging.getLogger("replay_hld_fallback")

FIELDNAMES = [
    "instance_id",
    "subset",
    "N",
    "M",
    "correlation",
    "f",
    "seed",
    "solver",
    "class_ordering",
    "n_iter",
    "alpha",
    "k",
    "lambda_max",
    "lambda_est",
    "fallback_equal_split",
    "phase1_wall_s",
    "phase2_wall_s",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("instances"),
        help="Instance archive root (default: instances).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest JSON (default: <archive>/MANIFEST.json).",
    )
    parser.add_argument(
        "--subset",
        default="test",
        choices=("test", "tuning", "all"),
        help="Manifest subset (default: test, matches the pinned archive).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hld_smac_best.json"),
        help="HLD calibration config (default: configs/hld_smac_best.json).",
    )
    parser.add_argument(
        "--class-ordering",
        choices=tuple(CLASS_ORDERINGS),
        default=DEFAULT_CLASS_ORDERING,
        help="Class-ordering strategy (default: sequential, paper default).",
    )
    parser.add_argument(
        "--lambda-max-override",
        type=float,
        default=None,
        help="Override lambda_max (default: use config payload).",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help="Skip instances with N > this value (smoke-test convenience).",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Cap the number of instances replayed (smoke-test convenience).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Where to write the replay results.",
    )
    return parser.parse_args(argv)


def _probe_one(
    *,
    instance: Any,
    settings: HldSettings,
    class_ordering: ClassOrdering,
    lambda_max_override: float | None,
) -> dict[str, float | int | bool]:
    """Run Phase-1 + Phase-2 estimate; return the fallback decision."""
    lambda_max = (
        float(lambda_max_override)
        if lambda_max_override is not None
        else float(settings.lambda_max)
        if settings.lambda_max
        else _instance_dependent_lambda_max(instance)
    )

    t0 = time.perf_counter()
    lambda_est, _trajectory = _phase1_binary_search(
        instance, lambda_max=lambda_max, n_iter=settings.n_iter
    )
    phase1_wall_s = time.perf_counter() - t0

    k = min(settings.k, instance.N)
    order = _class_order(instance, ordering=class_ordering, random_seed=None)
    batches = _split_classes(instance.N, k, order=order)

    t1 = time.perf_counter()
    _per_batch, fallback = _phase2_estimate(instance, batches=batches, lambda_est=lambda_est)
    phase2_wall_s = time.perf_counter() - t1

    return {
        "k": k,
        "lambda_max": lambda_max,
        "lambda_est": lambda_est,
        "fallback_equal_split": bool(fallback),
        "phase1_wall_s": phase1_wall_s,
        "phase2_wall_s": phase2_wall_s,
    }


def _row(
    entry: Any,
    settings: HldSettings,
    class_ordering: ClassOrdering,
    probe: dict[str, float | int | bool],
) -> dict[str, Any]:
    return {
        "instance_id": entry.rel_path,
        "subset": entry.subset,
        "N": entry.cell.get("N"),
        "M": entry.cell.get("M"),
        "correlation": entry.cell.get("correlation"),
        "f": entry.cell.get("f"),
        "seed": entry.seed,
        "solver": "hld",
        "class_ordering": class_ordering,
        "n_iter": settings.n_iter,
        "alpha": settings.alpha,
        "k": probe["k"],
        "lambda_max": probe["lambda_max"],
        "lambda_est": probe["lambda_est"],
        "fallback_equal_split": 1 if probe["fallback_equal_split"] else 0,
        "phase1_wall_s": f"{probe['phase1_wall_s']:.6f}",
        "phase2_wall_s": f"{probe['phase2_wall_s']:.6f}",
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    args = parse_args(argv)
    settings = load_hld_settings(args.config)

    entries = load_entries(
        archive_root=args.archive,
        manifest_path=args.manifest,
        subset=args.subset,
        max_n=args.max_n,
        max_instances=args.max_instances,
    )
    LOGGER.info("planned %d instances", len(entries))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_fallback = 0
    t0 = time.perf_counter()
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for i, entry in enumerate(entries, start=1):
            instance = load_instance(args.archive / entry.rel_path)
            probe = _probe_one(
                instance=instance,
                settings=settings,
                class_ordering=args.class_ordering,
                lambda_max_override=args.lambda_max_override,
            )
            n_fallback += int(probe["fallback_equal_split"])
            writer.writerow(_row(entry, settings, args.class_ordering, probe))
            if i % 200 == 0 or i == len(entries):
                elapsed = time.perf_counter() - t0
                LOGGER.info(
                    "replay progress %d/%d elapsed=%.1fs fallbacks=%d",
                    i,
                    len(entries),
                    elapsed,
                    n_fallback,
                )

    elapsed = time.perf_counter() - t0
    rate = (n_fallback / len(entries)) if entries else 0.0
    print(f"archive: {args.archive}")
    print(f"manifest: {args.manifest or args.archive / 'MANIFEST.json'}")
    print(f"subset: {args.subset}")
    print(f"config: {args.config}")
    print(f"class_ordering: {args.class_ordering}")
    print(f"out_csv: {args.out_csv}")
    print(f"n_instances: {len(entries)}")
    print(f"n_fallbacks: {n_fallback}")
    print(f"fallback_rate: {rate:.4f}")
    print(f"wall_time_s: {elapsed:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
