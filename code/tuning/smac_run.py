"""SMAC3-driven HLD parameter tuning (parameter-tuning spec §4.1, §4.2).

Exposes the HLD parameter quadruple ``(N_iter, alpha, K, lambda_max)``
as the target of a SMAC3 :class:`AlgorithmConfigurationFacade` over the
**tuning** subset of the instance archive. Per-evaluation cost is the
optimality gap relative to a HiGHS reference profit (cached once per
instance).

Outputs (default: ``tuning/smac_run/``):

- ``runhistory.json``  — full SMAC run history (auto-written by SMAC).
- ``scenario.json``    — SMAC scenario (auto-written by SMAC).
- ``incumbent.json``   — recommended configuration with bootstrap CI.
- ``reference_profits.json`` — HiGHS oracle profits per tuning instance.
- ``evaluations.csv``  — per-trial ``(config_id, instance, gap, time)``.

Usage
-----
Preview (existing N=200 archive, 80 tuning instances, ~30 trials, fast)::

    uv run python -m tuning.smac_run --preview --budget 30 --seed 7

Full (gated on the §2.2.1 archive being on disk)::

    uv run python -m tuning.smac_run --budget 5000 --seed 7

Design references
-----------------
- Parameter space and target metric: design D5.
- Tuning-only safety hook: ``instances.split.assert_tuning_only``.
- Bootstrap CI: ``tuning.bootstrap.bootstrap_mean_ci``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instances.io import load_instance
from instances.schema import InstanceModel
from instances.split import (
    DEFAULT_MASTER_SEED,
    DEFAULT_TUNING_RATIO,
    CellKey,
    assert_tuning_only,
)
from solvers import get_solver
from solvers.hld import HldAdapter

from tuning.bootstrap import bootstrap_mean_ci

LOGGER = logging.getLogger("tuning.smac_run")

REFERENCE_SOLVER = "highs"
DEFAULT_OUT_DIR = Path("tuning") / "smac_run"
DEFAULT_BUDGET = 5_000
DEFAULT_PREVIEW_BUDGET = 30
DEFAULT_SEED = 7
PARAM_SPACE = {
    "n_iter": {"low": 5, "high": 50, "default": 20},
    "alpha": {"low": 0.0, "high": 1.0, "default": 0.9},
    "k": {"low": 4, "high": 64, "default": 8},
    "lambda_max": {"low": 1.0, "high": 100.0, "default": 10.0},
}


# ---------------------------------------------------------------------------
# Manifest loading + tuning-subset selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuningInstance:
    """One tuning-subset instance with cached reference profit."""

    rel_path: str
    inst: InstanceModel
    ref_profit: int
    ref_time_s: float


@dataclass
class TuningArchive:
    """The set of tuning instances driving one SMAC campaign."""

    archive_root: Path
    items: list[TuningInstance]
    seeds_per_cell: dict[CellKey, list[int]]
    tuning_ratio: float = DEFAULT_TUNING_RATIO
    master_seed: int = DEFAULT_MASTER_SEED
    by_id: dict[str, TuningInstance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_id = {item.rel_path: item for item in self.items}

    @property
    def instance_ids(self) -> list[str]:
        return [item.rel_path for item in self.items]


def load_tuning_archive(
    *,
    archive_root: Path,
    manifest_path: Path | None = None,
    max_instances: int | None = None,
    reference_cache: Path | None = None,
    time_limit_s: float | None = None,
) -> TuningArchive:
    """Read manifest, load tuning-subset instances, populate HiGHS oracle."""
    archive_root = Path(archive_root)
    manifest_path = manifest_path or (archive_root / "MANIFEST.json")
    manifest = json.loads(manifest_path.read_text())

    seeds_per_cell: dict[CellKey, list[int]] = defaultdict(list)
    tuning_entries: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        cell = CellKey(
            N=entry["cell"]["N"],
            M=entry["cell"]["M"],
            correlation=entry["cell"]["correlation"],
            f=entry["cell"]["f"],
        )
        seeds_per_cell[cell].append(int(entry["seed"]))
        if entry.get("subset") == "tuning":
            tuning_entries.append(entry)

    if not tuning_entries:
        raise RuntimeError(
            f"No tuning-subset entries in manifest {manifest_path}; "
            "regenerate the archive or check the subset field."
        )

    tuning_entries.sort(key=lambda e: e["path"])
    if max_instances is not None:
        tuning_entries = tuning_entries[:max_instances]

    cache = _load_reference_cache(reference_cache) if reference_cache else {}
    items = _build_tuning_items(
        archive_root=archive_root,
        tuning_entries=tuning_entries,
        seeds_per_cell=seeds_per_cell,
        cache=cache,
        time_limit_s=time_limit_s,
        tuning_ratio=manifest.get("tuning_ratio", DEFAULT_TUNING_RATIO),
        master_seed=manifest.get("master_seed", DEFAULT_MASTER_SEED),
    )
    if reference_cache is not None:
        _save_reference_cache(reference_cache, items)

    return TuningArchive(
        archive_root=archive_root,
        items=items,
        seeds_per_cell=dict(seeds_per_cell),
        tuning_ratio=manifest.get("tuning_ratio", DEFAULT_TUNING_RATIO),
        master_seed=manifest.get("master_seed", DEFAULT_MASTER_SEED),
    )


def _build_tuning_items(
    *,
    archive_root: Path,
    tuning_entries: list[dict[str, Any]],
    seeds_per_cell: dict[CellKey, list[int]],
    cache: dict[str, dict[str, float]],
    time_limit_s: float | None,
    tuning_ratio: float,
    master_seed: int,
) -> list[TuningInstance]:
    ref_solver = get_solver(REFERENCE_SOLVER)
    items: list[TuningInstance] = []
    for entry in tuning_entries:
        path = archive_root / entry["path"]
        inst = load_instance(path)
        cell_seeds = sorted(seeds_per_cell[CellKey.from_instance(inst)])
        assert_tuning_only(
            inst,
            cell_seeds=cell_seeds,
            tuning_ratio=tuning_ratio,
            master_seed=master_seed,
        )
        cached = cache.get(entry["path"])
        if cached is not None:
            ref_profit = int(cached["profit"])
            ref_time = float(cached["time_s"])
        else:
            t0 = time.perf_counter()
            ref = ref_solver.solve(inst, time_limit_s=time_limit_s)
            ref_time = time.perf_counter() - t0
            ref_profit = int(ref.profit)
            LOGGER.info("reference[%s]: profit=%d time=%.2fs", entry["path"], ref_profit, ref_time)
        items.append(
            TuningInstance(
                rel_path=entry["path"],
                inst=inst,
                ref_profit=ref_profit,
                ref_time_s=ref_time,
            )
        )
    return items


def _load_reference_cache(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        LOGGER.warning("reference cache at %s is corrupt; recomputing", path)
        return {}


def _save_reference_cache(path: Path, items: Iterable[TuningInstance]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        item.rel_path: {"profit": item.ref_profit, "time_s": item.ref_time_s}
        for item in items
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# HLD evaluation (pure, SMAC-independent)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HldConfig:
    """Resolved HLD parameter point sampled from the SMAC config space."""

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


@dataclass(frozen=True)
class HldEvaluation:
    """Per-trial evaluation result."""

    instance_id: str
    config: HldConfig
    seed: int
    profit: int
    ref_profit: int
    optimality_gap: float
    wall_time_s: float


def evaluate_hld(
    item: TuningInstance,
    config: HldConfig,
    *,
    seed: int = 0,
    time_limit_s: float | None = None,
) -> HldEvaluation:
    """Run HLD with ``config`` on ``item`` and return the optimality gap.

    Gap is ``max(0, (ref - profit) / ref)``; clamped to zero so SMAC
    never sees a (numerically) negative cost from rounding.
    """
    solver = HldAdapter(
        n_iter=config.n_iter,
        alpha=config.alpha,
        k=config.k,
        lambda_max_override=config.lambda_max,
    )
    t0 = time.perf_counter()
    res = solver.solve(item.inst, time_limit_s=time_limit_s, random_seed=seed)
    wall = time.perf_counter() - t0
    if item.ref_profit <= 0:
        gap = 0.0
    else:
        gap = max(0.0, (item.ref_profit - res.profit) / item.ref_profit)
    return HldEvaluation(
        instance_id=item.rel_path,
        config=config,
        seed=seed,
        profit=int(res.profit),
        ref_profit=item.ref_profit,
        optimality_gap=float(gap),
        wall_time_s=float(wall),
    )


# ---------------------------------------------------------------------------
# SMAC wiring
# ---------------------------------------------------------------------------


def build_configspace() -> Any:
    """Define the four-dimensional HLD parameter space (lazy SMAC import)."""
    from ConfigSpace import ConfigurationSpace, Float, Integer

    ps = PARAM_SPACE
    cs = ConfigurationSpace()
    cs.add(Integer("n_iter", (ps["n_iter"]["low"], ps["n_iter"]["high"]),
                   default=ps["n_iter"]["default"]))
    cs.add(Float("alpha", (ps["alpha"]["low"], ps["alpha"]["high"]),
                 default=ps["alpha"]["default"]))
    cs.add(Integer("k", (ps["k"]["low"], ps["k"]["high"]),
                   default=ps["k"]["default"]))
    cs.add(Float("lambda_max", (ps["lambda_max"]["low"], ps["lambda_max"]["high"]),
                 default=ps["lambda_max"]["default"]))
    return cs


def run_smac_campaign(
    archive: TuningArchive,
    *,
    out_dir: Path,
    budget: int,
    seed: int,
    eval_time_limit_s: float | None = None,
    name: str = "hld_smac",
) -> tuple[Any, list[HldEvaluation]]:
    """Run the SMAC AlgorithmConfigurationFacade and return ``(incumbent, evaluations)``."""
    from smac import AlgorithmConfigurationFacade, Scenario  # lazy: SMAC has heavy import

    out_dir.mkdir(parents=True, exist_ok=True)
    cs = build_configspace()
    scenario = Scenario(
        configspace=cs,
        name=name,
        output_directory=out_dir,
        deterministic=True,
        n_trials=int(budget),
        instances=archive.instance_ids,
        seed=int(seed),
    )

    evaluations: list[HldEvaluation] = []

    def target_function(config: Any, instance: str, seed: int) -> float:
        item = archive.by_id[instance]
        assert_tuning_only(
            item.inst,
            cell_seeds=sorted(archive.seeds_per_cell[CellKey.from_instance(item.inst)]),
            tuning_ratio=archive.tuning_ratio,
            master_seed=archive.master_seed,
        )
        cfg = HldConfig.from_mapping(dict(config))
        ev = evaluate_hld(item, cfg, seed=int(seed), time_limit_s=eval_time_limit_s)
        evaluations.append(ev)
        return ev.optimality_gap

    smac = AlgorithmConfigurationFacade(scenario, target_function)
    incumbent = smac.optimize()
    return incumbent, evaluations


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_evaluations_csv(path: Path, evaluations: list[HldEvaluation]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance_id", "n_iter", "alpha", "k", "lambda_max",
        "seed", "profit", "ref_profit", "optimality_gap", "wall_time_s",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for ev in evaluations:
            w.writerow({
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
            })


def evaluate_incumbent_full(
    archive: TuningArchive,
    config: HldConfig,
    *,
    seed: int = 0,
    eval_time_limit_s: float | None = None,
) -> list[HldEvaluation]:
    """Re-evaluate the incumbent on every tuning instance for unbiased reporting."""
    return [
        evaluate_hld(item, config, seed=seed, time_limit_s=eval_time_limit_s)
        for item in archive.items
    ]


def write_incumbent_json(
    path: Path,
    *,
    config: HldConfig,
    evaluations: list[HldEvaluation],
    n_resamples: int,
    n_trials_total: int,
    smac_seed: int,
) -> None:
    gaps = [ev.optimality_gap for ev in evaluations]
    times = [ev.wall_time_s for ev in evaluations]
    gap_ci = bootstrap_mean_ci(gaps, n_resamples=n_resamples, random_seed=smac_seed)
    time_ci = bootstrap_mean_ci(times, n_resamples=n_resamples, random_seed=smac_seed)
    payload = {
        "config": {
            "N_iter": config.n_iter,
            "alpha": config.alpha,
            "K": config.k,
            "lambda_max": config.lambda_max,
        },
        "param_space": PARAM_SPACE,
        "smac": {
            "seed": smac_seed,
            "n_trials_total": int(n_trials_total),
            "n_evaluations_recorded": len(evaluations),
        },
        "tuning_subset_size": len(evaluations),
        "optimality_gap": gap_ci.as_dict(),
        "runtime_s": time_ci.as_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("instances"),
                   help="Root of the instance archive (default: instances/).")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Manifest path (default: <archive>/MANIFEST.json).")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Output directory (default: {DEFAULT_OUT_DIR}).")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help=f"Total SMAC trial budget (default: {DEFAULT_BUDGET}).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"SMAC random seed (default: {DEFAULT_SEED}).")
    p.add_argument("--max-instances", type=int, default=None,
                   help="Cap on tuning instances (default: all in subset).")
    p.add_argument("--eval-time-limit-s", type=float, default=None,
                   help="Per-trial HLD wall-clock cap (default: none).")
    p.add_argument("--ref-time-limit-s", type=float, default=None,
                   help="HiGHS oracle wall-clock cap per instance (default: none).")
    p.add_argument("--bootstrap-resamples", type=int, default=1000,
                   help="Bootstrap resamples for incumbent CI (default: 1000).")
    p.add_argument("--preview", action="store_true",
                   help="Use preview budget + write to <out-dir>/preview/.")
    return p.parse_args(argv)


def _resolve_out_dir(args: argparse.Namespace) -> tuple[Path, int]:
    if args.preview:
        budget = args.budget if args.budget != DEFAULT_BUDGET else DEFAULT_PREVIEW_BUDGET
        return args.out_dir / "preview", budget
    return args.out_dir, args.budget


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = parse_args(argv)
    out_dir, budget = _resolve_out_dir(args)

    LOGGER.info("Loading tuning archive from %s (manifest=%s)", args.archive, args.manifest)
    archive = load_tuning_archive(
        archive_root=args.archive,
        manifest_path=args.manifest,
        max_instances=args.max_instances,
        reference_cache=out_dir / "reference_profits.json",
        time_limit_s=args.ref_time_limit_s,
    )
    LOGGER.info("Tuning subset: %d instances across %d cells",
                len(archive.items), len(archive.seeds_per_cell))

    incumbent, evaluations = run_smac_campaign(
        archive,
        out_dir=out_dir,
        budget=budget,
        seed=args.seed,
        eval_time_limit_s=args.eval_time_limit_s,
    )
    config = HldConfig.from_mapping(dict(incumbent))
    LOGGER.info("Incumbent: n_iter=%d alpha=%.3f k=%d lambda_max=%.3f",
                config.n_iter, config.alpha, config.k, config.lambda_max)

    write_evaluations_csv(out_dir / "evaluations.csv", evaluations)
    LOGGER.info("Re-evaluating incumbent on full tuning subset for unbiased CI...")
    incumbent_evals = evaluate_incumbent_full(
        archive, config, seed=args.seed, eval_time_limit_s=args.eval_time_limit_s
    )
    write_incumbent_json(
        out_dir / "incumbent.json",
        config=config,
        evaluations=incumbent_evals,
        n_resamples=args.bootstrap_resamples,
        n_trials_total=len(evaluations),
        smac_seed=args.seed,
    )
    LOGGER.info("Wrote incumbent.json to %s", out_dir / "incumbent.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
