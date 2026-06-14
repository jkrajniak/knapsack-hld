#!/usr/bin/env python3
"""Re-tune HLD on a Pisinger-inclusive training distribution (Task R.7 §3.13).

This script is parallel to (and independent of) ``code/tuning/smac_run.py``.
Background: the original SMAC incumbent (`configs/hld_smac_best.json`,
λ_max=80.745) was tuned on the project's native instance distribution which
under-represents Pisinger weakly-correlated cells. The Pisinger validation
grid (`scripts/run_pisinger_validation.py`) showed HLD bleeds 5-9 pp of
optimality gap on type-2 large-n cells purely from λ_max miscalibration:
the diagnostic λ-sweep found best λ_max≈0.5 on those cells but ~2 on
small cells, well below the 80.745 incumbent.

This runner re-runs SMAC against on-the-fly Pisinger instances spanning
all six types so the resulting incumbent must compromise across the
distribution. Validation seeds (1..100) are excluded from training by
construction — training seeds live in [1001, 1012].

CLI shape mirrors `tuning.smac_run` so a `--preview` mode is available
for smoke testing on the local machine.

Outputs (under --out-dir):
- ``runhistory.json`` (auto), ``scenario.json`` (auto), SMAC working dir
- ``incumbent.json`` — best config + bootstrap CI of optimality gap
- ``evaluations.csv`` — every SMAC trial
- ``reference_profits.json`` — HiGHS oracle cache (resumable)
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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from instances.pisinger_generator import generate_pisinger_instance
from instances.schema import InstanceModel
from solvers import get_solver, validate_solution
from solvers.hld import HldAdapter

LOGGER = logging.getLogger("run_pisinger_smac")

DEFAULT_OUT_DIR = Path("tuning") / "pisinger_smac"
DEFAULT_BUDGET = 3000
DEFAULT_PREVIEW_BUDGET = 30
DEFAULT_SEED = 7
DEFAULT_TIME_LIMIT_S = 5.0
DEFAULT_REF_TIME_LIMIT_S = 30.0

# (type_id, k, n, r) — 24 cells spanning all six types × validation grid axes
DEFAULT_CELLS: list[tuple[int, int, int, int]] = [
    (t, k, n, 10000)
    for t in (1, 2, 3, 4, 5, 6)
    for k, n in ((10, 10), (10, 100), (100, 10), (100, 100))
]
DEFAULT_SEEDS = list(range(1001, 1013))  # 12 per cell, disjoint from validation 1..100

PARAM_SPACE = {
    "n_iter": {"low": 5, "high": 50, "default": 20},
    "alpha": {"low": 0.0, "high": 1.0, "default": 0.9},
    "k": {"low": 4, "high": 64, "default": 8},
    "lambda_max": {"low": 0.1, "high": 100.0, "default": 5.0},
}


@dataclass(frozen=True)
class TrainingItem:
    instance_id: str
    inst: InstanceModel
    ref_profit: int
    ref_time_s: float


@dataclass(frozen=True)
class HldConfig:
    n_iter: int
    alpha: float
    k: int
    lambda_max: float

    @classmethod
    def from_mapping(cls, m: dict[str, Any]) -> HldConfig:
        return cls(
            n_iter=int(m["n_iter"]),
            alpha=float(m["alpha"]),
            k=int(m["k"]),
            lambda_max=float(m["lambda_max"]),
        )


def _instance_id(type_id: int, k: int, n: int, r: int, seed: int) -> str:
    return f"pisinger_smac_t{type_id}_k{k}_n{n}_r{r}_s{seed}"


def _build_training_pool(
    *,
    cells: list[tuple[int, int, int, int]],
    seeds: list[int],
    ref_time_limit_s: float,
    cache_path: Path,
) -> list[TrainingItem]:
    """Generate Pisinger instances and resolve HiGHS reference profits.

    Reference cache is written incrementally so a partial run is recoverable.
    """
    cache: dict[str, dict[str, Any]] = (
        json.loads(cache_path.read_text()) if cache_path.exists() else {}
    )
    pool: list[TrainingItem] = []
    highs = get_solver("highs")
    total = len(cells) * len(seeds)
    t0 = time.perf_counter()

    for idx, ((t, k, n, r), seed) in enumerate(product(cells, seeds), start=1):
        iid = _instance_id(t, k, n, r, seed)
        inst = generate_pisinger_instance(n_classes=k, n_items=n, r=r, type_id=t, seed=seed)
        if (
            iid in cache
            and cache[iid].get("status") == "optimal"
            and cache[iid].get("profit", 0) > 0
        ):
            pool.append(
                TrainingItem(
                    instance_id=iid,
                    inst=inst,
                    ref_profit=int(cache[iid]["profit"]),
                    ref_time_s=float(cache[iid]["time_s"]),
                )
            )
            continue
        ts = time.perf_counter()
        ref = highs.solve(inst, time_limit_s=ref_time_limit_s)
        validate_solution(inst, ref)
        ref_t = time.perf_counter() - ts
        if str(ref.status) != "optimal" or ref.profit <= 0:
            LOGGER.warning("skipping %s (HiGHS status=%s profit=%d)", iid, ref.status, ref.profit)
            cache[iid] = {"status": str(ref.status), "profit": int(ref.profit), "time_s": ref_t}
            cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
            continue
        pool.append(
            TrainingItem(
                instance_id=iid,
                inst=inst,
                ref_profit=int(ref.profit),
                ref_time_s=ref_t,
            )
        )
        cache[iid] = {"status": "optimal", "profit": int(ref.profit), "time_s": ref_t}
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
        if idx % 20 == 0 or idx == total:
            LOGGER.info(
                "ref %d/%d valid=%d elapsed=%.1fs last=%s profit=%d in %.2fs",
                idx,
                total,
                len(pool),
                time.perf_counter() - t0,
                iid,
                ref.profit,
                ref_t,
            )
    return pool


def _build_configspace(param_space: dict[str, dict[str, float]]) -> Any:
    from ConfigSpace import ConfigurationSpace, Float, Integer

    cs = ConfigurationSpace()
    cs.add(
        Integer(
            "n_iter",
            (param_space["n_iter"]["low"], param_space["n_iter"]["high"]),
            default=param_space["n_iter"]["default"],
        )
    )
    cs.add(
        Float(
            "alpha",
            (param_space["alpha"]["low"], param_space["alpha"]["high"]),
            default=param_space["alpha"]["default"],
        )
    )
    cs.add(
        Integer(
            "k",
            (param_space["k"]["low"], param_space["k"]["high"]),
            default=param_space["k"]["default"],
        )
    )
    cs.add(
        Float(
            "lambda_max",
            (param_space["lambda_max"]["low"], param_space["lambda_max"]["high"]),
            default=param_space["lambda_max"]["default"],
        )
    )
    return cs


def _evaluate_hld(
    item: TrainingItem, cfg: HldConfig, *, seed: int, time_limit_s: float
) -> dict[str, Any]:
    solver = HldAdapter(
        n_iter=cfg.n_iter, alpha=cfg.alpha, k=cfg.k, lambda_max_override=cfg.lambda_max
    )
    t0 = time.perf_counter()
    res = solver.solve(item.inst, time_limit_s=time_limit_s, random_seed=seed)
    wall = time.perf_counter() - t0
    gap = max(0.0, (item.ref_profit - int(res.profit)) / item.ref_profit)
    return {
        "instance_id": item.instance_id,
        "n_iter": cfg.n_iter,
        "alpha": cfg.alpha,
        "k": cfg.k,
        "lambda_max": cfg.lambda_max,
        "seed": seed,
        "profit": int(res.profit),
        "ref_profit": item.ref_profit,
        "optimality_gap": float(gap),
        "wall_time_s": float(wall),
    }


def _bootstrap_mean(values: list[float], *, resamples: int, seed: int) -> dict[str, float]:
    import numpy as np

    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--eval-time-limit-s", type=float, default=DEFAULT_TIME_LIMIT_S)
    ap.add_argument("--ref-time-limit-s", type=float, default=DEFAULT_REF_TIME_LIMIT_S)
    ap.add_argument("--bootstrap-resamples", type=int, default=1000)
    ap.add_argument(
        "--preview",
        action="store_true",
        help="Use 30-trial preview budget; suffix out-dir with /preview/",
    )
    ap.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Cap on training instances (debugging only).",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )

    out_dir = args.out_dir / "preview" if args.preview else args.out_dir
    budget = DEFAULT_PREVIEW_BUDGET if args.preview else args.budget
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "reference_profits.json"

    LOGGER.info(
        "Building Pisinger training pool: %d cells × %d seeds = %d instances",
        len(DEFAULT_CELLS),
        len(DEFAULT_SEEDS),
        len(DEFAULT_CELLS) * len(DEFAULT_SEEDS),
    )
    pool = _build_training_pool(
        cells=DEFAULT_CELLS,
        seeds=DEFAULT_SEEDS,
        ref_time_limit_s=args.ref_time_limit_s,
        cache_path=cache_path,
    )
    if args.max_instances is not None:
        pool = pool[: args.max_instances]
    LOGGER.info("Training pool ready: %d valid instances", len(pool))
    by_id = {item.instance_id: item for item in pool}

    LOGGER.info("Param space: %s", PARAM_SPACE)
    LOGGER.info(
        "Launching SMAC: budget=%d trials, eval_time=%.1fs, seed=%d",
        budget,
        args.eval_time_limit_s,
        args.seed,
    )

    from smac import AlgorithmConfigurationFacade, Scenario  # heavy import deferred

    cs = _build_configspace(PARAM_SPACE)
    scenario = Scenario(
        configspace=cs,
        name="pisinger_smac",
        output_directory=out_dir,
        deterministic=True,
        n_trials=int(budget),
        instances=[item.instance_id for item in pool],
        seed=int(args.seed),
        n_workers=1,
    )

    evaluations: list[dict[str, Any]] = []

    def target_function(config: Any, instance: str, seed: int) -> tuple[float, dict[str, Any]]:
        item = by_id[instance]
        cfg = HldConfig.from_mapping(dict(config))
        ev = _evaluate_hld(item, cfg, seed=int(seed), time_limit_s=args.eval_time_limit_s)
        evaluations.append(ev)
        return ev["optimality_gap"], ev

    smac = AlgorithmConfigurationFacade(scenario, target_function)
    incumbent = smac.optimize()

    cfg = HldConfig.from_mapping(dict(incumbent))
    LOGGER.info(
        "Incumbent: n_iter=%d alpha=%.4f k=%d lambda_max=%.4f",
        cfg.n_iter,
        cfg.alpha,
        cfg.k,
        cfg.lambda_max,
    )

    csv_path = out_dir / "evaluations.csv"
    with csv_path.open("w", newline="") as fh:
        if evaluations:
            w = csv.DictWriter(fh, fieldnames=list(evaluations[0].keys()))
            w.writeheader()
            w.writerows(evaluations)
    LOGGER.info("wrote %d trial rows to %s", len(evaluations), csv_path)

    LOGGER.info("Re-evaluating incumbent across full pool for unbiased CI...")
    full = [
        _evaluate_hld(item, cfg, seed=args.seed, time_limit_s=args.eval_time_limit_s)
        for item in pool
    ]
    gaps = [e["optimality_gap"] for e in full]
    times = [e["wall_time_s"] for e in full]
    payload = {
        "config": {
            "N_iter": cfg.n_iter,
            "alpha": cfg.alpha,
            "K": cfg.k,
            "lambda_max": cfg.lambda_max,
        },
        "param_space": PARAM_SPACE,
        "smac": {"seed": args.seed, "n_trials_total": len(evaluations)},
        "training_pool_size": len(pool),
        "training_cells": [
            {"type_id": t, "k": k, "n": n, "r": r} for (t, k, n, r) in DEFAULT_CELLS
        ],
        "training_seeds": DEFAULT_SEEDS,
        "optimality_gap": _bootstrap_mean(gaps, resamples=args.bootstrap_resamples, seed=args.seed),
        "runtime_s": _bootstrap_mean(times, resamples=args.bootstrap_resamples, seed=args.seed),
    }
    inc_path = out_dir / "incumbent.json"
    inc_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    LOGGER.info("wrote incumbent to %s", inc_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
