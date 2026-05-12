"""HiGHS exact MILP adapter for Selective-MCKP.

Uses the official `highspy` Python API. We build the model column by
column so we don't pay the cost of materialising N*M Python objects per
constraint twice.
"""

from __future__ import annotations

import time
from typing import Any

import highspy
from instances.schema import InstanceModel

from solvers._milp_common import extract_selection, iter_pairs, make_result
from solvers.base import SolveResult, SolverStatus
from solvers.registry import register


class HighsAdapter:
    """HiGHS exact MILP solver for Selective-MCKP.

    Parameters
    ----------
    mip_rel_gap:
        Optional override for HiGHS's `mip_rel_gap` option. ``None`` keeps
        the HiGHS default (currently 1e-4 = 0.01%) which is what every
        out-of-the-box invocation has used historically. Pass a tighter
        value (e.g. ``1e-9``) when this adapter is used as an
        optimality-gap *reference* and the small default tolerance would
        otherwise produce visibly negative gaps for high-quality
        heuristics. The chosen tolerance is recorded in
        ``solver_metadata['mip_rel_gap_set']`` for traceability.
    threads:
        Optional override for HiGHS's internal thread count. This is useful
        when many independent HiGHS solves are executed in parallel; capping
        each solve to one thread avoids CPU oversubscription.
    """

    name: str = "highs"

    def __init__(self, *, mip_rel_gap: float | None = None, threads: int | None = None) -> None:
        if mip_rel_gap is not None and mip_rel_gap < 0:
            raise ValueError(f"mip_rel_gap must be non-negative, got {mip_rel_gap}")
        if threads is not None and threads < 1:
            raise ValueError(f"threads must be >= 1, got {threads}")
        self._mip_rel_gap = mip_rel_gap
        self._threads = threads

    def solve(
        self,
        instance: InstanceModel,
        *,
        time_limit_s: float | None = None,
        random_seed: int | None = None,
    ) -> SolveResult:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", "on")
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
        if time_limit_s is not None:
            h.setOptionValue("time_limit", float(time_limit_s))
        if random_seed is not None:
            h.setOptionValue("random_seed", int(random_seed) % (2**31 - 1))
        if self._mip_rel_gap is not None:
            h.setOptionValue("mip_rel_gap", float(self._mip_rel_gap))
        if self._threads is not None:
            h.setOptionValue("threads", int(self._threads))

        var_index = self._build(h, instance)
        t0 = time.perf_counter()
        h.run()
        wall = time.perf_counter() - t0

        info = h.getInfo()
        model_status = h.getModelStatus()
        status = _translate_status(h, model_status)

        if status is SolverStatus.INFEASIBLE:
            return make_result(
                profit=0,
                items_selected={i: None for i in range(instance.N)},
                total_cost=0,
                wall_time_s=wall,
                status=status,
                metadata={
                    "highs_status": str(model_status),
                    "mip_gap": None,
                    "mip_rel_gap_set": (
                        float(self._mip_rel_gap) if self._mip_rel_gap is not None else None
                    ),
                    "threads_set": int(self._threads) if self._threads is not None else None,
                },
            )

        sol = h.getSolution()
        col_value = sol.col_value

        def _value(i: int, j: int) -> float:
            return float(col_value[var_index[i, j]])

        sel, profit, cost = extract_selection(instance, _value)

        gap = getattr(info, "mip_gap", None)
        meta: dict[str, Any] = {
            "highs_status": str(model_status),
            "mip_gap": float(gap) if gap is not None else None,
            "mip_rel_gap_set": (
                float(self._mip_rel_gap) if self._mip_rel_gap is not None else None
            ),
            "threads_set": int(self._threads) if self._threads is not None else None,
            "objective_value": float(getattr(info, "objective_function_value", profit)),
            "simplex_iterations": int(getattr(info, "simplex_iteration_count", 0)),
            "mip_node_count": int(getattr(info, "mip_node_count", 0)),
        }
        return make_result(
            profit=profit,
            items_selected=sel,
            total_cost=cost,
            wall_time_s=wall,
            status=status,
            metadata=meta,
        )

    @staticmethod
    def _build(h: highspy.Highs, instance: InstanceModel) -> dict[tuple[int, int], int]:
        """Add variables + constraints; return `(i, j) -> column index`."""
        var_index: dict[tuple[int, int], int] = {}
        for col, (i, j, p, _c) in enumerate(iter_pairs(instance)):
            h.addCol(float(p), 0.0, 1.0, 0, [], [])
            var_index[i, j] = col
        h.changeColsIntegrality(
            len(var_index),
            list(range(len(var_index))),
            [highspy.HighsVarType.kInteger] * len(var_index),
        )

        for i in range(instance.N):
            cols = [var_index[i, j] for j in range(instance.M)]
            vals = [1.0] * instance.M
            h.addRow(0.0, 1.0, instance.M, cols, vals)

        cols = list(var_index.values())
        vals = [float(instance.items[i][j][1]) for (i, j) in var_index]
        h.addRow(0.0, float(instance.B), len(cols), cols, vals)
        return var_index


def _translate_status(h: highspy.Highs, model_status) -> SolverStatus:
    """Map a HiGHS model status onto our SolverStatus enum."""
    s = h.modelStatusToString(model_status).lower() if model_status is not None else ""
    if "optimal" in s:
        return SolverStatus.OPTIMAL
    if "time" in s:
        return SolverStatus.TIMEOUT
    if "infeasible" in s:
        return SolverStatus.INFEASIBLE
    if any(t in s for t in ("primal feasible", "feasible")):
        return SolverStatus.FEASIBLE
    return SolverStatus.ERROR


@register("highs")
def _factory() -> HighsAdapter:
    return HighsAdapter()
