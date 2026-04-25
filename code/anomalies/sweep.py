"""Anomaly subset generator + HLD/HiGHS sweep driver (spec §4.3.1).

The sweep is fully deterministic:

- Instance generation reuses the canonical ``instances.generate_instance``
  with the same ``(N, M, correlation, f, seed)`` keys as the main
  archive. Generated files are written to
  ``instances/anomalies/...`` so subsequent calls reuse them instead of
  regenerating.
- HLD is the default (``alpha=0.9, K=8``, instance-dependent
  ``lambda_max``) so that the sweep isolates the effect of ``N_iter``.
  The full per-iteration ``solver_metadata`` is preserved in the
  output, which is what :mod:`anomalies.analyse` consumes.
- HiGHS solves each anomaly instance once (cached in memory) to provide
  a reference profit for computing the optimality gap.

Output: one JSON record per ``(instance, n_iter)`` pair, written to
``results/anomalies/sweep.jsonl`` (line-delimited JSON for streaming).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from instances.schema import InstanceModel
from solvers.base import SolveResult, SolverStatus
from solvers.hld import HldAdapter
from solvers.registry import get_solver

from instances import (
    CorrelationKind,
    generate_instance,
    instance_id,
    load_instance,
    save_instance,
)

LOG = logging.getLogger("anomalies.sweep")

DEFAULT_N_ITER_GRID: tuple[int, ...] = tuple(range(1, 26))
DEFAULT_SEEDS: tuple[int, ...] = (0, 7, 42)
DEFAULT_CELL: dict[str, Any] = {
    "N": 10_000,
    "M": 10,
    "correlation": "weakly",
    "f": 0.5,
}


@dataclass(frozen=True)
class AnomalyInstance:
    """One instance in the anomaly subset, with its on-disk path."""

    inst: InstanceModel
    path: Path
    inst_id: str


@dataclass(frozen=True)
class SweepRecord:
    """One ``(instance, n_iter)`` HLD evaluation record."""

    inst_id: str
    n_iter: int
    hld_profit: int
    opt_profit: int
    optimality_gap: float
    hld_wall_s: float
    opt_wall_s: float
    solver_metadata: dict[str, Any]
    budget: int
    opt_status: str = "unknown"
    opt_metadata: dict[str, Any] | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "inst_id": self.inst_id,
            "n_iter": self.n_iter,
            "hld_profit": self.hld_profit,
            "opt_profit": self.opt_profit,
            "optimality_gap": self.optimality_gap,
            "hld_wall_s": self.hld_wall_s,
            "opt_wall_s": self.opt_wall_s,
            "solver_metadata": self.solver_metadata,
            "budget": self.budget,
            "opt_status": self.opt_status,
            "opt_metadata": self.opt_metadata or {},
        }


def ensure_anomaly_subset(
    *,
    archive_root: Path,
    cell: dict[str, Any] = DEFAULT_CELL,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[AnomalyInstance]:
    """Load (or generate-and-save) the deterministic anomaly subset."""
    cell_dir = archive_root / "anomalies" / _cell_dirname(cell)
    cell_dir.mkdir(parents=True, exist_ok=True)
    out: list[AnomalyInstance] = []
    for seed in seeds:
        iid = instance_id(seed=seed, **cell)
        path = cell_dir / f"{iid}.json.gz"
        if path.exists():
            inst = load_instance(path)
        else:
            LOG.info("generating anomaly instance %s", iid)
            inst = generate_instance(seed=seed, **cell)
            save_instance(inst, path)
        out.append(AnomalyInstance(inst=inst, path=path, inst_id=iid))
    return out


def run_one(
    *,
    item: AnomalyInstance,
    n_iter: int,
    opt_profit: int,
    opt_wall_s: float,
    opt_status: str = "unknown",
    opt_metadata: dict[str, Any] | None = None,
    sub_solver: str = "highs",
    eval_time_limit_s: float | None = None,
) -> SweepRecord:
    """Run HLD with the requested ``n_iter`` and produce a SweepRecord."""
    hld = HldAdapter(n_iter=n_iter, sub_solver=sub_solver)
    t0 = time.perf_counter()
    res: SolveResult = hld.solve(item.inst, time_limit_s=eval_time_limit_s)
    wall = time.perf_counter() - t0
    gap = (
        (opt_profit - res.profit) / opt_profit
        if opt_profit > 0
        else float("nan")
    )
    return SweepRecord(
        inst_id=item.inst_id,
        n_iter=n_iter,
        hld_profit=int(res.profit),
        opt_profit=int(opt_profit),
        optimality_gap=float(gap),
        hld_wall_s=float(wall),
        opt_wall_s=float(opt_wall_s),
        solver_metadata=dict(res.solver_metadata),
        budget=int(item.inst.B),
        opt_status=opt_status,
        opt_metadata=dict(opt_metadata) if opt_metadata else {},
    )


def run_sweep(
    *,
    items: list[AnomalyInstance],
    n_iter_grid: tuple[int, ...] = DEFAULT_N_ITER_GRID,
    sub_solver: str = "highs",
    reference_solver: str = "highs",
    reference_time_limit_s: float | None = 600.0,
    eval_time_limit_s: float | None = None,
    out_path: Path | None = None,
) -> list[SweepRecord]:
    """Run the full sweep and (optionally) stream JSONL to ``out_path``."""
    refs: dict[str, tuple[int, float, str, dict[str, Any]]] = {}
    ref = get_solver(reference_solver)
    for item in items:
        LOG.info("solving %s with reference solver %s", item.inst_id, reference_solver)
        rt0 = time.perf_counter()
        ref_res = ref.solve(item.inst, time_limit_s=reference_time_limit_s)
        ref_wall = time.perf_counter() - rt0
        ref_status = (
            ref_res.status.value
            if isinstance(ref_res.status, SolverStatus)
            else str(ref_res.status)
        )
        refs[item.inst_id] = (
            int(ref_res.profit),
            ref_wall,
            ref_status,
            dict(ref_res.solver_metadata or {}),
        )
        LOG.info(
            "  -> %s profit=%d status=%s wall=%.1fs",
            item.inst_id,
            ref_res.profit,
            ref_status,
            ref_wall,
        )

    records: list[SweepRecord] = []
    out_fh = out_path.open("w") if out_path is not None else None
    try:
        for item in items:
            opt_profit, opt_wall, opt_status, opt_meta = refs[item.inst_id]
            for n_iter in n_iter_grid:
                rec = run_one(
                    item=item,
                    n_iter=n_iter,
                    opt_profit=opt_profit,
                    opt_wall_s=opt_wall,
                    opt_status=opt_status,
                    opt_metadata=opt_meta,
                    sub_solver=sub_solver,
                    eval_time_limit_s=eval_time_limit_s,
                )
                records.append(rec)
                if out_fh is not None:
                    out_fh.write(json.dumps(rec.as_json()) + "\n")
                    out_fh.flush()
                LOG.info(
                    "sweep %s n_iter=%d gap=%.4f hld_wall=%.2fs",
                    item.inst_id,
                    n_iter,
                    rec.optimality_gap,
                    rec.hld_wall_s,
                )
    finally:
        if out_fh is not None:
            out_fh.close()
    return records


def _cell_dirname(cell: dict[str, Any]) -> str:
    correlation = CorrelationKind(cell["correlation"]).value
    return f"{correlation}/N{cell['N']}_M{cell['M']}_f{cell['f']:0.3f}"


def load_sweep(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL sweep file back into raw record dicts."""
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
